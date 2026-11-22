# SparseMind Runtime (AMEFORGE)

Lanceur universel, hors-ligne, pour les modèles **SparseMind** d'AMEFORGE au
format **ONNX** (SAM-G, gc_editor, cofos, …). Un seul fichier, aucune dépendance
à PyTorch, aucun code d'architecture : il charge un `.onnx` et génère.

## Installation
```bash
git clone https://github.com/amforge1/sparsemind-runtime
cd sparsemind-runtime
pip install -r requirements.txt
```

## Utilisation
Télécharge un modèle depuis Hugging Face (ex. `AMFORGE/gc_editor`) — tu obtiens
`model.onnx`, un tokenizer (ex. `gearcut_tok.model`) et `config.json` — puis :
```bash
python run.py --model model.onnx --tokenizer gearcut_tok.model \
              --prompt "clip the first 5 seconds" --max-new 128
```
Déterministe par défaut (`--temp 0`). Échantillonnage : `--temp 0.7 --top-k 40`.

## Modèles compatibles
Tout modèle AMEFORGE publié au format ONNX. Les poids et l'architecture restent
la propriété d'AMEFORGE ; ce dépôt ne contient que le lanceur.

## Licence du lanceur
MIT (le lanceur uniquement). Les modèles ont leur propre licence sur Hugging Face.
