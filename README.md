# Person Detection – yolo26n + CPU Stress Test

Détection de personnes (et des autres objets COCO) avec **yolo26n** (Ultralytics) sur des vidéos,
plus des outils pour mesurer les performances CPU (stress test et benchmark multi-flux).

## Structure

```
person_detection/
├── input_videos/    vidéos à analyser (.mp4, .avi, .mov, .mkv…)
├── output_videos/   vidéos annotées
├── reports/         rapports JSON / CSV / PNG
├── models/          yolo26n.pt (téléchargé au premier lancement)
├── common.py        modèle + fonctions partagées
├── app.py           interface graphique
├── detect.py        détection simple
├── stress_test.py   stress test multi-cœurs
├── benchmark.py     nombre de flux supportés en parallèle
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Interface graphique

```bash
python app.py
```

- **Détection** : choisir une vidéo, régler la confiance et la taille d'image, lancer. Affichage en direct avec statistiques ; la vidéo annotée est enregistrée dans `output_videos/`.
- **Stress test CPU** : choisir workers, threads, mode et durée. Usage CPU par cœur et logs en direct.

### Détection

```bash
python detect.py                   # personnes + tous les objets
python detect.py --persons-only    # personnes uniquement
python detect.py --conf 0.4        # seuil de confiance plus strict
```

Sortie : `output_videos/<nom>_detected.mp4`. Personnes en vert, autres objets en couleur, HUD en haut à gauche (personnes, objets, temps d'inférence).

### Stress test CPU

Lance plusieurs processus yolo26n en parallèle (chacun épinglé sur un cœur), affiche l'usage CPU/RAM chaque seconde et enregistre un résumé dans `reports/`.

```bash
python stress_test.py                                   # 1 worker par vidéo, toutes en même temps
python stress_test.py --streams 8                       # simule 8 caméras + mosaïque
python stress_test.py --mode split                      # 1 worker par cœur, file d'attente
python stress_test.py --mode replicate --duration 300   # charge max pendant 5 min
python stress_test.py --workers 2 --threads 2
```

| Option | Description |
|---|---|
| `--mode parallel\|split\|replicate` | `parallel` (défaut) : toutes les vidéos en même temps, 1 worker chacune · `split` : 1 worker par cœur, les vidéos attendent leur tour · `replicate` : chaque worker traite toutes les vidéos |
| `--workers N` | nombre de processus (ignoré en `parallel`) |
| `--threads N` | threads torch par processus (défaut : 1) |
| `--streams N` | nombre total de flux en dupliquant les vidéos (`_cam1` … `_camN`) |
| `--duration S` | durée max en secondes (en `replicate`, les vidéos bouclent) |
| `--no-save` | ne pas écrire les vidéos de sortie |
| `--no-mosaic` | ne pas créer `output_videos/mosaic_<date>.mp4` |
| `--no-pin` | ne pas fixer les processus sur des cœurs |
| `--persons-only` | personnes uniquement |
| `--imgsz`, `--conf` | taille d'entrée et seuil de confiance |

### Benchmark multi-flux

Mesure combien de vidéos peuvent être traitées en parallèle (1 processus par flux, N = 1, 2, 3…).

```bash
python benchmark.py                         # imgsz 640
python benchmark.py --target-fps 5          # temps réel = 5 FPS par flux
python benchmark.py --imgsz 320 640         # compare deux tailles
python benchmark.py --streams 1 2 4 --duration 30
```

Mesures par palier : FPS par flux (moyen / min), FPS total, latence (moyenne / p95), CPU, RAM.
Arrêt automatique si la RAM devient insuffisante (`--max-ram`, 92 % par défaut) ou si le débit baisse deux fois de suite.

Résumé : nombre max de flux en temps réel, meilleur débit total, capacité théorique, RAM par flux.
Fichiers : `reports/benchmark_<date>.json`, `.csv`, `.png`.

## Remarques

- Chaque worker charge son propre modèle : environ **440 Mo de RAM** par processus. Réduire `--workers` si la RAM sature.
- En `parallel` avec plus de vidéos que de cœurs, plusieurs vidéos partagent un cœur.
- En `split` avec moins de vidéos que de cœurs, des cœurs restent inactifs : utiliser `replicate` pour tout charger.
- Comparer `--workers N --threads 1` et `--workers 1 --threads N` pour trouver la meilleure configuration.
