# Medical GPT Training - MedQuAD Dataset
# MedQuAD: Medical Question Answering Dataset from trusted medical sources
# Contains 47,457 medical Q&A pairs from NIH, CDC, and other organizations

from datasets import load_dataset
import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from contextlib import nullcontext

# ============================================================================
# STEP 1: Load MedQuAD Dataset
# ============================================================================

print("Loading MedQuAD Medical Q&A Dataset...")
print("Source: Trusted medical organizations (NIH, CDC, etc.)\n")

try:
    # Try loading from HuggingFace
    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    print(f"✓ MedQuAD loaded successfully!")
except:
    # Alternative: try different naming
    try:
        ds = load_dataset("medquad", split="train")
        print(f"✓ MedQuAD loaded successfully!")
    except:
        print("⚠ Could not load MedQuAD from HuggingFace.")
        print("\nAlternative loading method:")
        print("1. Download from: https://github.com/abachaa/MedQuAD")
        print("2. Or use: !git clone https://github.com/abachaa/MedQuAD.git")
        print("\nFalling back to demo dataset...")
        ds = load_dataset("gamino/wiki_medical_terms", split="train")

print(f"\n{'='*70}")
print(f"Dataset Statistics:")
print(f"  Total samples: {len(ds):,}")
print(f"  Dataset features: {ds.features}")
print(f"{'='*70}\n")

# Display sample to understand structure
print("Sample from dataset:")
print("-" * 70)
if len(ds) > 0:
    sample = ds[0]
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"{key}: {value[:200]}...")
        else:
            print(f"{key}: {value}")
    print("-" * 70)
    print()
    
    # Show what the processed text looks like
    print("Testing text extraction on first sample:")
    print("-" * 70)
    test_text = format_medquad_text(sample)
    print(f"Extracted text ({len(test_text)} chars):")
    print(test_text[:500] if len(test_text) > 500 else test_text)
    print("-" * 70)
    print()
else:
    print("⚠ WARNING: Dataset appears to be empty!")
    print("-" * 70)
    print()

# ============================================================================
# STEP 2: Tokenization for MedQuAD Format
# ============================================================================

enc = tiktoken.get_encoding("gpt2")

def format_medquad_text(example):
    """
    Format MedQuAD examples into training text.
    MedQuAD typically has: question, answer, focus, and source fields
    """
    # Try different possible field names in MedQuAD
    question = ""
    answer = ""
    
    # Common field variations
    question_fields = ['question', 'Question', 'qtype', 'query']
    answer_fields = ['answer', 'Answer', 'response', 'text']
    
    # Extract question
    for field in question_fields:
        if field in example and example[field]:
            question = str(example[field]).strip()
            break
    
    # Extract answer
    for field in answer_fields:
        if field in example and example[field]:
            answer = str(example[field]).strip()
            break
    
    # Fallback: use all available text
    if not question and not answer:
        all_text = " ".join([str(v) for v in example.values() if isinstance(v, str)])
        return all_text
    
    # Format as Q&A conversation
    if question and answer:
        formatted_text = f"Question: {question}\n\nAnswer: {answer}"
    elif question:
        formatted_text = f"Question: {question}"
    elif answer:
        formatted_text = answer
    else:
        formatted_text = str(example)
    
    return formatted_text

def process(example):
    """Process MedQuAD examples into tokens"""
    try:
        # Format the text
        text = format_medquad_text(example)
        
        # Quality filters - LESS STRICT to avoid filtering everything
        if len(text) < 20:  # Reduced from 50
            return {'ids': [], 'len': 0}
        
        if len(text) > 15000:  # Truncate very long texts
            text = text[:15000]
        
        # Tokenize
        ids = enc.encode_ordinary(text)
        
        # Skip if too few tokens - LESS STRICT
        if len(ids) < 5:  # Reduced from 20
            return {'ids': ids, 'len': len(ids)}
        
        return {'ids': ids, 'len': len(ids)}
    
    except Exception as e:
        # Return empty on error
        return {'ids': [], 'len': 0}

# ============================================================================
# STEP 3: Create Training Data Files
# ============================================================================

if not os.path.exists("train.bin"):
    print("Processing MedQuAD dataset...")
    print("Formatting Q&A pairs and tokenizing...\n")
    
    # Tokenize the dataset
    tokenized = ds.map(
        process,
        remove_columns=ds.column_names,
        desc="Processing medical Q&A pairs",
        num_proc=4,
        batch_size=500,
    )
    
    # Filter out empty examples
    print("Filtering valid examples...")
    tokenized = tokenized.filter(lambda x: x['len'] > 0, num_proc=4)
    
    # CHECK IF DATASET IS EMPTY
    if len(tokenized) == 0:
        raise ValueError("ERROR: All examples were filtered out! Dataset is empty. Check your data format.")
    
    print(f"✓ Valid examples: {len(tokenized):,}")
    
    # CHECK MINIMUM SIZE
    if len(tokenized) < 100:
        print(f"⚠ WARNING: Dataset very small ({len(tokenized)} examples)")
        print("Consider using a larger dataset or checking data loading")
    
    # Split into train/validation (90/10)
    print("Creating train/validation split...")
    
    # Ensure we have enough data to split
    if len(tokenized) < 10:
        print("⚠ Dataset too small to split. Using all data for training.")
        train_data = tokenized
        val_data = tokenized  # Use same data for validation (not ideal but prevents crash)
    else:
        split_dataset = tokenized.train_test_split(test_size=0.1, seed=42)
        train_data = split_dataset['train']
        val_data = split_dataset['test']
    
    print(f"✓ Training samples: {len(train_data):,}")
    print(f"✓ Validation samples: {len(val_data):,}")
    
    # Calculate dataset statistics
    train_tokens = np.sum(train_data['len'], dtype=np.uint64)
    val_tokens = np.sum(val_data['len'], dtype=np.uint64)
    
    # CHECK FOR ZERO TOKENS
    if train_tokens == 0:
        raise ValueError("ERROR: Training data has 0 tokens! Check your text extraction logic.")
    
    print(f"\nDataset Token Statistics:")
    print(f"  Training tokens: {train_tokens:,} (~{train_tokens*2/1e6:.1f} MB)")
    print(f"  Validation tokens: {val_tokens:,} (~{val_tokens*2/1e6:.1f} MB)")
    print(f"  Total tokens: {train_tokens + val_tokens:,}")
    print()
    
    # Create binary files
    print("Creating memory-mapped binary files...")
    for split_name, dset in [('train', train_data), ('validation', val_data)]:
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        
        # SAFETY CHECK
        if arr_len == 0:
            print(f"⚠ WARNING: {split_name} has 0 tokens, skipping file creation")
            continue
            
        filename = f'{split_name}.bin'
        dtype = np.uint16
        
        print(f"\nWriting {split_name}.bin...")
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # Use adaptive batch count based on dataset size
        total_batches = min(512, max(32, len(dset) // 100))
        
        # SAFETY: Ensure at least 1 batch
        if total_batches == 0:
            total_batches = 1
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
        print(f"✓ {filename} created ({arr_len:,} tokens)")
    
    print(f"\n{'='*70}")
    print("✓ Dataset preparation complete!")
    print(f"{'='*70}\n")

# ============================================================================
# STEP 4: Data Loading Function
# ============================================================================

def get_batch(split, block_size, batch_size, device, device_type):
    """Load a batch of data"""
    filename = 'train.bin' if split == 'train' else 'validation.bin'
    
    if not os.path.exists(filename):
        filename = 'train.bin'
    
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    if len(data) <= block_size:
        raise ValueError(f"Dataset too small! Only {len(data)} tokens.")
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ============================================================================
# STEP 5: Model Architecture
# ============================================================================

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                              dropout_p=self.attn_dropout.p if self.training else 0.0, 
                                              is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate medical Q&A responses"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============================================================================
# STEP 6: Model Configuration (Optimized for MedQuAD Q&A)
# ============================================================================

config = GPTConfig(
    vocab_size=50257,
    block_size=384,      # Good size for Q&A pairs
    n_layer=10,          # Deep enough for medical reasoning
    n_head=10,
    n_embd=640,          # Medium-large model
    dropout=0.12,        # Moderate dropout for Q&A
    bias=True
)

# Training hyperparameters
learning_rate = 3e-4
max_iters = 25000        # Sufficient for MedQuAD size
warmup_steps = 1000
min_lr = 1e-5
eval_iters = 300
batch_size = 12
block_size = 384
gradient_accumulation_steps = 12  # Effective batch = 144

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print(f"{'='*70}")
print(f"Training Configuration:")
print(f"  Device: {device}")
print(f"  Precision: {dtype}")
print(f"  Context length: {block_size}")
print(f"  Batch size: {batch_size}")
print(f"  Gradient accumulation: {gradient_accumulation_steps}")
print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
print(f"  Learning rate: {learning_rate}")
print(f"  Max iterations: {max_iters:,}")
print(f"{'='*70}\n")

# Initialize model
torch.manual_seed(42)
model = GPT(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized:")
print(f"  Parameters: {total_params/1e6:.2f}M")
print(f"  Model size: ~{total_params * 2 / 1e9:.2f} GB (fp16)")
print()

# ============================================================================
# STEP 7: Optimizer and Scheduler
# ============================================================================

from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    betas=(0.9, 0.95), 
    weight_decay=0.1, 
    eps=1e-8
)

scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# ============================================================================
# STEP 8: Training Functions
# ============================================================================

def estimate_loss(model, eval_iters_count, ctx, block_size, batch_size, device, device_type):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters_count)
            for k in range(eval_iters_count):
                X, Y = get_batch(split, block_size, batch_size, device, device_type)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

# ============================================================================
# STEP 9: Training Loop
# ============================================================================

best_val_loss = float('inf')
best_model_path = "best_medquad_model.pt"
train_loss_list, validation_loss_list = [], []

print("Starting training on MedQuAD dataset...")
print("Training medical Q&A model...\n")
print("=" * 70)

for epoch in tqdm(range(max_iters), desc="Training"):
    # Evaluation
    if epoch % eval_iters == 0:
        losses = estimate_loss(model, 80, ctx, block_size, batch_size, device, device_type)
        
        print(f"\n{'='*70}")
        print(f"Iteration {epoch:5d}/{max_iters}")
        print(f"  Train Loss: {losses['train']:.4f}")
        print(f"  Val Loss:   {losses['val']:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        if train_loss_list:
            print(f"  Best Val:   {best_val_loss:.4f}")
        print(f"{'='*70}")
        
        train_loss_list.append(losses['train'])
        validation_loss_list.append(losses['val'])
        
        # Save best model
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': losses['val'],
                'config': config,
            }, best_model_path)
            print(f"✓ New best model saved! (val_loss: {best_val_loss:.4f})\n")
    
    # Training step
    X, y = get_batch("train", block_size, batch_size, device, device_type)
    
    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
    
    scaler.scale(loss).backward()
    
    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    scheduler.step()

print("\n" + "=" * 70)
print("✓ Training completed!")
print(f"✓ Best validation loss: {best_val_loss:.4f}")
print(f"✓ Model saved to: {best_model_path}")
print("=" * 70)

# ============================================================================
# STEP 10: Generate Medical Q&A Samples
# ============================================================================

print("\n\nGenerating Medical Q&A Samples...\n")
model.eval()

checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Medical question prompts
test_prompts = [
    "Question: What are the symptoms of diabetes?\n\nAnswer:",
    "Question: How is hypertension treated?\n\nAnswer:",
    "Question: What causes asthma?\n\nAnswer:",
    "Question: What are the risk factors for heart disease?\n\nAnswer:",
    "Question: How can I prevent the flu?\n\nAnswer:",
]

print("=" * 70)
print("MEDICAL Q&A GENERATION TEST")
print("=" * 70)

for i, prompt in enumerate(test_prompts, 1):
    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    generated = model.generate(x, max_new_tokens=200, temperature=0.7, top_k=50)
    generated_text = enc.decode(generated[0].tolist())
    
    print(f"\n[Sample {i}]")
    print("-" * 70)
    print(generated_text)
    print("=" * 70)

# ============================================================================
# STEP 11: Plot Training Curves
# ============================================================================

import matplotlib.pyplot as plt

train_loss_np = [loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in train_loss_list]
val_loss_np = [loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in validation_loss_list]

plt.figure(figsize=(12, 6))
plt.plot(train_loss_np, 'g-', label='Training Loss', linewidth=2.5, alpha=0.8)
plt.plot(val_loss_np, 'r-', label='Validation Loss', linewidth=2.5, alpha=0.8)
plt.xlabel(f'Evaluation Steps (every {eval_iters} iterations)', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.title('MedQuAD Medical Q&A Model Training', fontsize=15, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('medquad_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*70}")
print("TRAINING SUMMARY")
print(f"{'='*70}")
print(f"Model: Medical Q&A GPT (MedQuAD)")
print(f"Parameters: {total_params/1e6:.2f}M")
print(f"Training iterations: {max_iters:,}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final train loss: {train_loss_list[-1]:.4f}")
print(f"Final val loss: {val_loss_list[-1]:.4f}")
print(f"\nModel saved to: {best_model_path}")
print(f"Training curves: medquad_training_curves.png")
print(f"{'='*70}")
