import torch


def normalize(samples):
    """Normalize samples by dividing by the norm taken over all samples.
    
    Args:
        samples (~torch.Tensor): Tensor of shape :math:`(*, N, F)` where
            :math:`*` means any number of additional dimensions,
            :math:`N` is the number of samples and :math:`F` is the number
            of features per sample.
    
    Returns:
        ~torch.Tensor: Normalized tensor of same shape as :attr:`samples`.
    """
    return samples / samples.norm(dim=-2, keepdim=True)


def cosine_similarity_matrix(samples, eps=1e-8):
    """Calculate the cosine similarity between all sample pairs.
    
    Args:
        samples (~torch.Tensor): Tensor of shape :math:`(*, N, F)` where
            :math:`*` means any number of additional dimensions,
            :math:`N` is the number of samples and :math:`F` is the number
            of features per sample.
    
    Returns:
        ~torch.Tensor: Cosine similarity matrix of shape :math:`(*, N, N)`
        where :math:`*` and math:`N` are the same as for the input.
    """
    assert samples.dim() >= 2, \
        'Shape of input should be (*, num_samples, num_features)'
    w = samples.norm(dim=-1, keepdim=True)
    return samples.matmul(samples.transpose(-1, -2)) / (w * w.transpose(-1, -2)).clamp(min=eps)


def bmv(matrix, vector):
    """Batched matrix-vector multiplication.
    
    Args:
        matrix (~torch.Tensor): Tensor of shape :math:`(*, N, M)`
            where :math:`*` means any number of additional dimensions
            and :math:`N`, :math:`M` are arbitrary sizes.
        vector (~torch.Tensor): Tensor of shape :math:`(*, M)`
            where :math:`*` is the same number and sizes of additional
            dimensions as for :attr:`matrix` and :math:`M` is the same
            size as for :attr:`matrix`.
    
    Returns:
        ~torch.Tensor: The results of the matrix-vector multiplication
        of shape :math:`(*, N)` where :math:`*` and :math:`N` are the
        same as for the input arguments.
    """
    return torch.matmul(matrix, vector.unsqueeze(-1)).squeeze(-1)


def bdot(vector_a, vector_b):
    """Batched inner product of vectors.
    
    Args:
        vector_a (~torch.Tensor): Tensor of shape :math:`(*, N)`
            where :math:`*` means any number of additional dimensions
            and :math:`N`is of arbitrary size.
        vector_b (~torch.Tensor): Tensor that can be broadcasted to
            shape of :attr:`vector_a`.
    
    Returns:
        ~torch.Tensor: The results of the inner product
        of shape :math:`(*)` where :math:`*` is the same number
        of dimensions and sizes as for the input arguments.
    """
    return torch.sum(vector_a * vector_b, dim=-1)
    

def embed_padded_states(states, mask=None, context_size=2):
    """Embed inner states of a BERT-style model that may be padded.
    
    Args:
        states (~torch.Tensor): Tensor of shape (batch, tokens, layers, features).
        mask (~torch.BoolTensor, optional): Optional mask of shape (batch, tokens)
        context_size (int): As defined in the paper, how many layers are we looking
            at "above" and "below" the "current" layer. Default value is 2.
    
    Returns:
        (~torch.Tensor): Embeddings of shape (batch, features)
    """
    if states.numel() == 0:
        return states.new()
    assert states.dim() == 4, \
        'Expected states of shape (batch, tokens, layers, features). ' + \
        'Got {}'.format(tuple(states.size()))
    bsz, tsz, lsz, fsz = states.size()
    
    if mask is None:
        return embed_states(states=states, context_size=context_size)
    
    mask_shape = tuple(mask.size())
    assert mask_shape == (bsz, tsz), \
        'Mask should be of shape (batch, tokens), ' + \
        'expected {} '.format((bsz, tsz)) + \
        'so as to match the shape of `states` but got ' + \
        '{}.'.format(mask_shape)
    num_pads = tsz - mask.sum(dim=-1)
    
    unique = num_pads.unique()
        
    if unique.size(0) == 1:
        return embed_states(
            states=states[mask].view(bsz, -1, lsz, fsz),
            context_size=context_size
        )

    embeddings = states.new_empty(bsz, fsz)

    for num_pad in unique:
        batch_mask = num_pads.eq(num_pad)
        batch_states = states[batch_mask].view(-1, tsz, lsz, fsz)
        if num_pad > 0:
            batch_states = batch_states[mask[batch_mask]].view(-1, tsz - num_pad, lsz, fsz)
        embeddings[batch_mask] = embed_states(
            states=batch_states,
            context_size=context_size
        )

    return embeddings

    
def embed_states(states, context_size=2):
    """Embed inner states of a BERT-style model.
    
    Args:
        states (~torch.Tensor): Tensor of shape (batch, tokens, layers, features).
        context_size (int): As defined in the paper, how many layers are we looking
            at "above" and "below" the "current" layer. Default value is 2.
    
    Returns:
        (~torch.Tensor): Embeddings of shape (batch, features)
    """
    if states.numel() == 0:
        return states.new()
    assert states.dim() == 4, \
        'Expected states of shape (batch, tokens, layers, features). ' + \
        'Got {}'.format(tuple(states.size()))
    bsz, tsz, lsz, fsz = states.size()
    
    alpha_alignment_list = []
    alpha_novelty_list = []

    for layer_index in range(lsz):
        
        left_window = states[..., layer_index - context_size: layer_index, :]
        right_window = states[..., layer_index + 1: layer_index + context_size + 1, :]
        
        window_matrix = torch.cat(
            (
                left_window,
                right_window,
                states[..., layer_index, :].unsqueeze(-2)
            ),
            dim=-2
        )
        
        Q, R = torch.qr(window_matrix.transpose(-1, -2))
        q = Q[..., -1]
        r = R[..., -1]
        
        alpha_alignment = bdot(
            torch.mean(
                normalize(
                    R[..., :-1, :-1]
                ),
                dim=-1
            ),
            R[..., :-1, -1]
        )
        alpha_alignment = alpha_alignment / r[..., :-1].norm(dim=-1)
        alpha_alignment = 1 / ((2 * window_matrix.size(-2)) * alpha_alignment)
        alpha_alignment_list.append(alpha_alignment)
        
        alpha_novelty = r[..., -1].abs() / r.norm(dim=-1)
        alpha_novelty_list.append(alpha_novelty)
        
    alpha_alignment = torch.stack(alpha_alignment_list, dim=-1)
    alpha_novelty = torch.stack(alpha_novelty_list, dim=-1)

    alpha_alignment = alpha_alignment / alpha_alignment.sum(dim=-1, keepdim=True)
    alpha_novelty = alpha_novelty / alpha_novelty.sum(dim=-1, keepdim=True)

    alpha = alpha_alignment + alpha_novelty

    alpha = alpha / alpha.sum(dim=-1, keepdim=True)
    
    # bsz, tsz, fsz
    token_embeddings = bmv(states.transpose(-1, -2), alpha)
    
    # bsz, tsz, lsz, lsz
    similarity_matrix = cosine_similarity_matrix(states)
    
    # bsz, tsz
    token_vars = similarity_matrix.diagonal(offset=-1, dim1=-2, dim2=-1).var(dim=-1)
    token_vars = token_vars / token_vars.sum(dim=-1, keepdim=True)
    
    # bsz, fsz
    return bmv(token_embeddings.transpose(-1, -2), token_vars)
