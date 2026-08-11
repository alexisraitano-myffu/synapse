# Politique de confidentialité de Synapse

**Dernière mise à jour : 10 août 2026**

Synapse est une mémoire personnelle. Le principe qui gouverne toute l'application
est simple : **tes captures t'appartiennent et restent sur tes appareils**. Ce
document dit exactement ce qui est traité, ce qui sort, et vers qui.

## Qui est responsable

Synapse est édité par **Alexis Raitano**, responsable du traitement au sens du
règlement européen sur la protection des données.

Contact : **alexis.raitano@myffu.fr**

## Aucun compte, aucune inscription

Synapse ne demande ni nom, ni adresse e-mail, ni numéro de téléphone. Il n'existe
aucun compte utilisateur, donc aucun profil te concernant n'est constitué.

## Ce que l'application traite

- **Le contenu de tes captures** : le texte que tu écris ou dictes.
- **Ce qui en est dérivé** : les personnes, projets, faits, tâches et dates
  extraits de ces textes, ainsi que les liens entre eux.
- **Tes réglages** : langue, thème, préférences de notification, et un
  identifiant aléatoire généré par l'application pour distinguer tes propres
  appareils entre eux. Cet identifiant n'a aucun lien avec ton téléphone, ton
  compte Google ou un identifiant publicitaire.

Tout cela est stocké **sur ton appareil**, et sur ton ordinateur si tu utilises
l'application de bureau.

## Ce qui quitte tes appareils, et pourquoi

### Le classement de tes captures

Pour transformer une capture en mémoire organisée, son texte est envoyé à un
modèle de langage, **Claude, édité par Anthropic**. C'est le seul moment où le
contenu d'une capture quitte tes appareils, et cela n'a lieu que pour rendre ce
service. La connexion est chiffrée (HTTPS).

Deux cas selon ta configuration :

- **Ta propre clé Anthropic** : l'appel part directement vers Anthropic. Le
  traitement de tes données par Anthropic relève alors de leurs conditions.
- **Un token de beta fermée** (`syn-fuel-…`), *pendant la phase de test fermé
  uniquement* : l'appel transite alors par un relais technique hébergé chez
  Cloudflare et opéré par l'éditeur. Sa seule fonction est de porter la clé
  Anthropic à ta place, pour que tu n'aies pas à en fournir une. La requête, donc
  le texte de la capture, passe par ce relais avant d'être transmise à Anthropic.

  À la fin de la phase de test, plus aucun token de ce type n'est délivré : le
  relais sort du chemin et cette section disparaîtra de la présente politique.

Selon les conditions d'Anthropic pour son API, les contenus envoyés ne servent
pas à entraîner leurs modèles et ne sont conservés que le temps nécessaire au
service et au respect de leurs obligations. Leurs conditions font foi sur ce
point : elles sont publiées sur anthropic.com.

### La synchronisation entre tes appareils

Ton téléphone et ton ordinateur échangent directement, sur ton réseau Wi-Fi,
sans passer par aucun serveur. Cet échange est protégé par un jeton d'accès
propre à ton installation, mais **il n'est pas chiffré** : sur un réseau que tu
ne contrôles pas, quelqu'un qui écoute ce réseau pourrait en lire le contenu.
Utilise Synapse sur un réseau de confiance, pas sur un Wi-Fi public.

### La sauvegarde Android

Si la sauvegarde automatique d'Android est activée sur ton téléphone (réglage du
système, pas de l'application), les données de Synapse en font partie et sont
copiées vers ton propre espace Google Drive. Depuis Android 9, ces sauvegardes
sont chiffrées avec le code de verrouillage de ton téléphone, et Google ne peut
pas les lire. Tu peux désactiver cette sauvegarde dans les réglages Android.

### Le téléchargement du modèle

Au premier usage, l'application télécharge un modèle de calcul depuis Hugging
Face. Cette requête ne transporte aucune de tes données : elle ne fait que
récupérer un fichier public.

## Ce que l'application ne fait pas

- **Aucune mesure d'audience, aucun traceur, aucun outil de suivi de plantage.**
  Aucun code tiers de ce type n'est présent dans l'application.
- **Aucune publicité**, et donc aucun identifiant publicitaire.
- **Aucun enregistrement sonore.** La dictée utilise la reconnaissance vocale de
  ton téléphone, qui rend du texte ; l'application ne reçoit, ne stocke et
  n'envoie jamais de son.
- **Aucune photo.** La caméra ne sert qu'à scanner le QR code d'appairage de tes
  appareils ; l'image est analysée sur le moment et n'est ni conservée ni
  transmise.
- **Aucune vente ni cession** de tes données à qui que ce soit.

## Les autorisations demandées

| Autorisation | Pourquoi |
|---|---|
| Internet | Classer tes captures, et synchroniser avec ton ordinateur |
| Micro | Dicter une capture, via la reconnaissance vocale du téléphone |
| Caméra | Scanner le QR code qui appaire tes appareils entre eux |
| Notifications | Te rappeler tes tâches, tes échéances et le digest |

Micro et caméra sont demandés au moment où tu t'en sers, et refuser l'un ou
l'autre laisse le reste de l'application pleinement utilisable.

## Ta clé de classement

Ta clé Anthropic, ou ton token de beta, est conservée dans le magasin sécurisé du
système (Android Keystore), chiffrée par une clé qui ne peut pas sortir de
l'appareil. Elle n'est jamais synchronisée vers tes autres appareils, sauf si tu
choisis explicitement de la partager au moment d'appairer un appareil.

## Combien de temps, et comment supprimer

Tes données restent aussi longtemps que tu les gardes : il n'y a aucune durée
imposée, parce qu'aucun serveur ne les détient.

- Supprimer une note ou une fiche se fait dans l'application.
- Désinstaller l'application efface les données qu'elle détient sur cet appareil.
- Sur l'ordinateur, désinstaller l'application de bureau retire son moteur ; le
  dossier de données peut être supprimé à la main.
- Si la sauvegarde Android est activée, pense à la supprimer aussi, depuis les
  réglages Google de ton téléphone.

L'éditeur ne détient aucune copie de tes captures et ne peut donc ni te les
restituer ni les supprimer à ta place.

## Tes droits

Le règlement européen te donne un droit d'accès, de rectification, d'effacement,
de limitation, d'opposition et de portabilité. Dans le cas de Synapse, ces droits
s'exercent directement dans l'application, puisque tu détiens tes données. Pour
toute question, écris à **alexis.raitano@myffu.fr**.

Tu peux aussi introduire une réclamation auprès de la CNIL (cnil.fr).

## Enfants

Synapse ne s'adresse pas aux mineurs et n'est pas conçu pour eux.

## Modifications

Toute évolution de cette politique sera publiée à cette adresse, avec une date de
mise à jour. Un changement qui élargirait ce qui sort de tes appareils sera
annoncé dans l'application.
