# 🧠 NanoGPT: A Tiny Yet Powerful Transformer Language Model

## 📘 Project Overview

This project implements a **bigram-level character language model** using a **Transformer architecture** from scratch in PyTorch — closely following Karpathy’s excellent video lectures.

The model takes raw text, learns character-level dependencies, and generates new text one character at a time. While tiny in scale, it reproduces the core ideas behind large transformer-based models like **GPT-2/GPT-3**.

> 🔬 This project is meant purely for **learning purposes**, helping understand how large language models work at their core.

---

## 🔧 Key Techniques & Concepts Used

### ✅ 1. **Tokenization & Vocabulary**

* The text is tokenized at the **character level**, with each unique character assigned an integer.
* Two mappings are created:

  * `stoi`: character → integer
  * `itos`: integer → character

```python
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

---

### ✅ 2. **Batching & Data Preparation**

* The model is trained on **random contiguous sequences** (`block_size`) from the dataset.
* Each input tensor `x` has shape `(batch_size, block_size)`, and the corresponding target `y` is the next character.

```python
x = torch.stack([data[i : i + block_size] for i in ix])
y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
```

---

### ✅ 3. **Model Architecture**

#### 📌 **Embedding Layer**

* Converts token IDs into continuous vectors via:

  * **Token Embedding Table**
  * **Position Embedding Table**

```python
token_emb = self.token_embedding_table(idx)
pos_emb = self.position_embedding_table(torch.arange(T, device=device))
x = token_emb + pos_emb
```

---

#### 📌 **Transformer Blocks**

Each block contains:

* **Multi-Head Self-Attention (MHSA)**
* **FeedForward MLP**
* **Layer Normalization**
* **Residual Connections**

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

---

#### 📌 **Self-Attention with Causal Masking**

* Enforces **auto-regressive property**: each token can only attend to itself and previous tokens
* Achieved with a lower-triangular attention mask (`tril`)

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

---

#### 📌 **Multi-Head Attention**

* Splits embeddings into `n_head` chunks and applies attention in parallel
* Outputs are concatenated and projected

```python
self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
```

---

#### 📌 **FeedForward Network**

* A simple 2-layer MLP with a ReLU activation and dropout

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),
    nn.Dropout(dropout)
)
```

---

### ✅ 4. **Loss Function**

* Uses **cross-entropy loss** between predicted logits and target characters
* The logits are reshaped for classification across vocabulary

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

---

### ✅ 5. **Training Loop**

* Optimized with **AdamW** optimizer
* Trains for a specified number of iterations
* Periodically logs **train/validation loss**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

---

### ✅ 6. **Text Generation (Sampling)**

* Starts with a small context (e.g., a zero token)
* Iteratively predicts the next token using the model, sampling from the probability distribution

```python
for _ in range(max_new_tokens):
    logits = model(idx_cond)[0]
    probs = F.softmax(logits[:, -1, :], dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)
```

---

## 📈 Sample Output

> After training on `tinyshakespeare.txt`, here’s a snippet of generated text:

```text
FLORIZEL:
A greater favour.

PERCNIUS:
A general stay's business: is't love?

CLARENCE:
It is, by God's name: good comfors! good my lord,
Which men in most wife lady kiss down,
That will brings mine, and if it it receive
To meet on his chastity wag'd with dainty,
Cutiolanus, fit in their enemies.
I'll have to save, the queen lives: and to you,
Holp the king thee, your lordship.
```
---

## 📚 What You’ll Learn by Doing This

✅ How GPT-style language models work internally

✅ The role of embeddings, attention, and feedforward networks

✅ How to train a transformer using only PyTorch

✅ Token sampling and auto-regressive text generation

✅ Why causal masking matters in language modeling

✅ How optimizers like AdamW work in practice

---

## 🖥️ How to Run This

```bash
# Install dependencies
pip install torch numpy

# Run training
python bigram_gpt.py
```

---

## 📌 Acknowledgments

This project is a **learning replication** of:

* 📺 [Andrej Karpathy’s NG video lectures](https://github.com/karpathy/ng-video-lecture)
* 🧠 Inspired by the core ideas in OpenAI’s GPT models
