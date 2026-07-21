import torch

from app import (
    ImageGenerator, ImageDiscriminator,
    TextGenerator, TextDiscriminator,
    IMG_LATENT, TXT_LATENT, VOCAB_SIZE, SEQ_LEN,
    WORDS, W2I, I2W, make_real_batch,
)


def test_image_generator_output_shape():
    G = ImageGenerator()
    z = torch.randn(5, IMG_LATENT)
    out = G(z)
    assert out.shape == (5, 1, 28, 28)
    assert out.min() >= -1.0001 and out.max() <= 1.0001  # tanh range


def test_image_discriminator_output_shape():
    D = ImageDiscriminator()
    x = torch.randn(5, 1, 28, 28)
    out = D(x)
    assert out.shape == (5, 1)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid range


def test_image_gan_gradient_flows_generator_to_discriminator():
    G, D = ImageGenerator(), ImageDiscriminator()
    z = torch.randn(4, IMG_LATENT)
    fake = G(z)
    score = D(fake)
    loss = score.mean()
    loss.backward()
    grad_norms = [p.grad.abs().sum().item() for p in G.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(g > 0 for g in grad_norms)


def test_text_generator_output_shape():
    G = TextGenerator()
    z = torch.randn(3, TXT_LATENT)
    logits = G(z)
    assert logits.shape == (3, SEQ_LEN, VOCAB_SIZE)


def test_text_generator_gumbel_sample_is_one_hot_ish():
    G = TextGenerator()
    z = torch.randn(2, TXT_LATENT)
    sample = G.sample_gumbel(z, tau=1.0, hard=True)
    assert sample.shape == (2, SEQ_LEN, VOCAB_SIZE)
    # hard=True -> forward pass is one-hot per position
    sums = sample.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_text_discriminator_accepts_both_token_ids_and_soft_vectors():
    D = TextDiscriminator()
    real = torch.randint(0, VOCAB_SIZE, (3, SEQ_LEN))
    out_real = D(real)
    assert out_real.shape == (3, 1)

    G = TextGenerator()
    z = torch.randn(3, TXT_LATENT)
    fake = G.sample_gumbel(z, tau=1.0, hard=True)
    out_fake = D(fake)
    assert out_fake.shape == (3, 1)


def test_text_gan_gradient_flows_through_gumbel_softmax_to_generator():
    G, D = TextGenerator(), TextDiscriminator()
    z = torch.randn(4, TXT_LATENT)
    fake = G.sample_gumbel(z, tau=1.0, hard=True)  # not detached
    score = D(fake)
    score.mean().backward()
    grad_norms = [p.grad.abs().sum().item() for p in G.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(g > 0 for g in grad_norms)


def test_vocab_is_consistent():
    assert len(WORDS) == VOCAB_SIZE
    assert all(W2I[I2W[i]] == i for i in range(0, VOCAB_SIZE, 137))  # spot-check
    assert "<pad>" in W2I and "<unk>" in W2I


def test_make_real_batch_shape_and_padding():
    batch = make_real_batch(6)
    assert batch.shape == (6, SEQ_LEN)
    assert batch.dtype == torch.long
    # every row should be a valid sequence of known vocab ids
    assert (batch >= 0).all() and (batch < VOCAB_SIZE).all()
