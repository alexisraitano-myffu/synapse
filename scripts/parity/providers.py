"""SYN-171 — appeler un modèle candidat, quel que soit son runtime.

Deux dialectes suffisent aujourd'hui :

  * `anthropic:<model>` — la référence (Haiku), via `anthropic_client.get_client()`,
    donc le seam fuel-proxy de SYN-105 marche aussi pour un token `syn-fuel-`.
  * `ollama:<model>`    — tout ce qui tourne en local (Qwen, Llama, Gemma…).

Toute réponse est ramenée à la MÊME forme (`Reply`), pour que le scoring ne
sache pas d'où elle vient. C'est la leçon de SYN-150 côté core : normaliser au
plus près du réseau, et laisser le reste du code provider-agnostique.

⚠️ Le piège Ollama qui invalide silencieusement une mesure : `num_ctx` vaut
2048 par défaut selon les modèles. Notre prompt classifieur en fait ~4 500 —
Ollama tronque alors le DÉBUT du prompt sans le dire, et le modèle paraît
mauvais alors qu'il n'a jamais reçu les règles. On fixe donc `num_ctx`
explicitement ET on relit `prompt_eval_count` pour vérifier que le modèle a bien
avalé ce qu'on lui a envoyé (`Reply.prompt_tokens`).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

OLLAMA_URL = "http://localhost:11434/api/chat"
# Assez large pour le classifieur (~4 500 tokens) + la capture + la sortie JSON.
# Volontairement pas énorme : un num_ctx géant réserve du KV-cache pour rien et
# fausse la mesure d'empreinte mémoire (leçon SYN-154, cf. maxNumTokens 8192→6144).
DEFAULT_NUM_CTX = 8192


@dataclass
class Reply:
    """Réponse normalisée, identique quel que soit le provider."""

    text: str
    #  "stop" (fini) | "max_tokens" (tronqué) | autre chose (anormal)
    stop_reason: str | None
    latency_s: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"

    @property
    def ok(self) -> bool:
        return self.error is None and not self.truncated and bool(self.text.strip())


def parse_spec(spec: str) -> tuple[str, str]:
    """`ollama:qwen2.5:3b` → ('ollama', 'qwen2.5:3b'). Le modèle peut contenir ':'."""
    if ":" not in spec:
        raise ValueError(f"spec de modèle invalide : {spec!r} (attendu 'provider:model')")
    provider, model = spec.split(":", 1)
    if provider not in ("anthropic", "ollama"):
        raise ValueError(f"provider inconnu : {provider!r}")
    return provider, model


def call(spec: str, system_blocks: list[str], user: str, max_tokens: int,
         num_ctx: int = DEFAULT_NUM_CTX, schema: dict | None = None,
         temperature: float = 0.0) -> Reply:
    """Un appel, un `Reply`. Ne lève jamais : une panne est une donnée de mesure.

    `schema` (JSON Schema) active le décodage contraint quand le runtime le
    supporte. Ollama : passé dans `format`. Anthropic : **ignoré** — l'API n'a pas
    d'équivalent sur `messages.create`, et prétendre le contraire fausserait la
    comparaison. Une mesure contrainte ne se compare donc qu'à une autre mesure
    contrainte du même côté.
    """
    provider, model = parse_spec(spec)
    try:
        if provider == "anthropic":
            return _call_anthropic(model, system_blocks, user, max_tokens, temperature)
        return _call_ollama(model, system_blocks, user, max_tokens, num_ctx, schema,
                            temperature)
    except Exception as exc:  # noqa: BLE001 — un modèle qui casse EST un résultat
        return Reply(text="", stop_reason=None, latency_s=0.0,
                     error=f"{type(exc).__name__}: {exc}")


def _call_anthropic(model: str, system_blocks: list[str], user: str,
                    max_tokens: int, temperature: float = 0.0) -> Reply:
    from anthropic_client import get_client

    # Le premier bloc porte le cache : c'est ce que fait le core (le classifieur
    # est stable, les blocs vocab/projets bougent). Reproduire la vraie forme,
    # pas une forme simplifiée, sinon on ne mesure pas ce qui tourne en prod.
    blocks = [{"type": "text", "text": system_blocks[0],
               "cache_control": {"type": "ephemeral"}}]
    blocks += [{"type": "text", "text": b} for b in system_blocks[1:]]

    # ⚠️ Sans température explicite, l'API échantillonne au défaut (1.0) et la
    # mesure devient irreproductible. Constaté le 2026-08-20 : sur DEUX passes du
    # même modèle sur le même texte, 2 cas sur 12 divergent, dont une bascule de
    # branche (« Nouveau projet : rénovation » produit une note, puis plus rien).
    # Le harnais fixait déjà 0 côté Ollama : la comparaison Haiku-vs-local était
    # donc faussée, un côté déterministe et l'autre non.
    # NB : le core, LUI, ne fixe pas la température — cette valeur mesure le
    # prompt, pas la variance de production. Passer --temperature 1.0 pour ça.
    t0 = time.time()
    msg = get_client().messages.create(
        model=model, max_tokens=max_tokens, system=blocks, temperature=temperature,
        messages=[{"role": "user", "content": user}],
    )
    text = msg.content[0].text if msg.content else ""
    # ⚠️ `input_tokens` EXCLUT ce qui a été servi par le cache de prompt : sur un
    # deuxième appel, le classifieur (~4 500 tokens) bascule dans
    # `cache_read_input_tokens` et `input_tokens` retombe à ~200. Compter la seule
    # valeur brute ferait croire que le modèle n'a pas reçu le prompt — le gate a
    # justement rendu ce faux NO-GO à sa première exécution.
    usage = msg.usage
    prompt_tokens = (usage.input_tokens
                     + (getattr(usage, "cache_read_input_tokens", 0) or 0)
                     + (getattr(usage, "cache_creation_input_tokens", 0) or 0))
    return Reply(
        text=text, stop_reason=msg.stop_reason, latency_s=round(time.time() - t0, 2),
        prompt_tokens=prompt_tokens, output_tokens=usage.output_tokens,
        extra={"uncached_input_tokens": usage.input_tokens},
    )


def _call_ollama(model: str, system_blocks: list[str], user: str,
                 max_tokens: int, num_ctx: int, schema: dict | None = None,
                 temperature: float = 0.0) -> Reply:
    # Ollama n'a pas de blocs système multiples ni de cache_control : on aplatit
    # avec le même séparateur que le core (\n\n) pour que le TEXTE reçu par le
    # modèle local soit identique à celui reçu par Haiku.
    system = "\n\n".join(system_blocks)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens,
                    "num_ctx": num_ctx},
    }
    if schema is not None:
        # Décodage contraint : le sampler n'accepte que les continuations valides
        # au regard du schéma. Rend les valeurs hors énumération impossibles.
        payload["format"] = schema
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=900).read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama injoignable sur {OLLAMA_URL} — lancer `ollama serve` ({exc})"
        ) from exc
    done = resp.get("done_reason")
    return Reply(
        text=resp.get("message", {}).get("content", "") or "",
        # `length` = budget de sortie épuisé : c'est le `max_tokens` d'Anthropic.
        stop_reason="max_tokens" if done == "length" else done,
        latency_s=round(time.time() - t0, 2),
        prompt_tokens=resp.get("prompt_eval_count"),
        output_tokens=resp.get("eval_count"),
        extra={"eval_s": round(resp.get("eval_duration", 0) / 1e9, 2),
               "num_ctx": num_ctx},
    )
