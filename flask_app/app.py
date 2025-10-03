from flask import Flask, request, render_template, Response, stream_with_context
import torch
from model import BigramLanguageModel
import torch.nn.functional as F

app = Flask(__name__)

# ----------------------------
# Load checkpoint
checkpoint = torch.load("flask_app/shakespeare_model_clean.pth", map_location="cpu")

batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ----------------------------

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Reconstruct the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_route():
    data = request.get_json()
    prompt = data.get("prompt", "")
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    max_tokens = 500

    def generate_streaming(idx, max_tokens):

        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            yield itos[idx_next.item()]

    return Response(stream_with_context(generate_streaming(idx, max_tokens)), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # for local use
