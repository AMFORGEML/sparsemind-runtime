#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — Lanceur générique pour les modèles SparseMind au format ONNX (AMEFORGE).

Un seul script pour TOUS les modèles : SAM-G, gc_editor, cofos, etc.
Il ne contient AUCUNE logique d'architecture — il charge un graphe .onnx compilé
et fait la boucle de génération par-dessus. C'est sûr à mettre en open source.

Dépendances minimales (pas de PyTorch) :
    pip install onnxruntime numpy sentencepiece

Exemple :
    python run.py --model model.onnx --tokenizer gearcut_tok.model \
                  --prompt "clip the first 5 seconds" --max-new 128

Tokenizer accepté :
    * SentencePiece : fichier .model  (--tokenizer tok.model)
    * HF tokenizers : dossier ou tokenizer.json (--tokenizer ./  ou tokenizer.json)
                      (nécessite alors `pip install tokenizers`)
"""
import argparse, json, os, sys
import numpy as np
import onnxruntime as ort


# ----------------------------- tokenizer -----------------------------
class SPTokenizer:
    def __init__(self, path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=path)
    def encode(self, s): return self.sp.encode(s, out_type=int)
    def decode(self, ids): return self.sp.decode(ids)

class HFTokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        self.tk = Tokenizer.from_file(path)
    def encode(self, s): return self.tk.encode(s).ids
    def decode(self, ids): return self.tk.decode(ids)

def load_tokenizer(path):
    if path.endswith(".model"):
        return SPTokenizer(path)
    if path.endswith(".json") or os.path.isdir(path):
        return HFTokenizer(path)
    # dernier recours : tenter SentencePiece
    return SPTokenizer(path)


# ----------------------------- sampling -----------------------------
def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def sample_next(logits, temp, top_k):
    logits = logits.astype(np.float64)
    if temp and temp > 0:
        logits = logits / temp
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        thresh = np.sort(logits)[-k]
        logits = np.where(logits < thresh, -np.inf, logits)
    if not temp or temp <= 0:
        return int(np.argmax(logits))          # greedy
    p = softmax(logits)
    return int(np.random.choice(len(p), p=p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.onnx")
    ap.add_argument("--tokenizer", required=True, help=".model SentencePiece ou tokenizer HF")
    ap.add_argument("--config", default="config.json", help="pour eos_id / max_seq_len (optionnel)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (déterministe)")
    ap.add_argument("--top-k", type=int, default=0)
    args = ap.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        try: cfg = json.load(open(args.config))
        except Exception: cfg = {}
    eos_id   = cfg.get("eos_id", 2)
    max_seq  = cfg.get("max_seq_len", 1024)

    tok  = load_tokenizer(args.tokenizer)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    ids = tok.encode(args.prompt)
    for _ in range(args.max_new):
        window = ids[-max_seq:]
        inp = np.array([window], dtype=np.int64)
        logits = sess.run(["logits"], {"input_ids": inp})[0]   # [1, T, vocab]
        nxt = sample_next(logits[0, -1, :], args.temp, args.top_k)
        ids.append(nxt)
        if nxt == eos_id:
            break

    print(tok.decode(ids))


if __name__ == "__main__":
    main()
