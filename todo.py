import torch
from config import config
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
_config = config()
def evaluate(golden_list, predict_list):
    nb_i = 0
    length = len(golden_list)
    FN,FP,TP = 0,0,0
    for i in range(length):
        ws = len(golden_list[i])
        golden_list_i = golden_list[i]
        predict_list_i = predict_list[i]
        k = 0
        while (k<ws):
            i_k = k
            if golden_list_i[k] == 'O':
                k += 1
                continue
            else:
                suffix = golden_list_i[k][2:]
                g_k = k
                if k < ws -1:
                    while (golden_list_i[k+1][2:] == suffix):
                        k += 1
                        if golden_list_i[k][0] == 'B':
                            k -= 1
                            break
                        if k == ws -1:
                            break
                    g_k = k
                    if k < ws -1:
                        while (predict_list_i[k+1][2:] == suffix):
                            k += 1
                            if predict_list_i[k][0] == 'B': 
                                k -= 1
                                break
                            if k == ws -1:
                                break
                if golden_list_i[i_k:k+1] != predict_list_i[i_k:k+1]:
                    FN += 1
                else:
                    TP += 1
                k = g_k + 1
        k = 0
        while (k<ws):
            if predict_list_i[k] == 'O':
                k += 1
                continue
            else:
                i_k = k
                suffix = predict_list_i[k][2:]
                g_k = k
                if k < ws -1:
                    while (predict_list_i[k+1][2:] == suffix):
                        k += 1
                        if predict_list_i[k][0] == 'B':
                            k -= 1
                            break
                        if k == ws -1:
                            break
                    g_k = k
                    if k < ws -1:
                        while (golden_list_i[k+1][2:] == suffix):
                            k += 1
                            if golden_list_i[k][0] == 'B':
                                k -= 1
                                break
                            if k == ws -1:
                                break
                if golden_list_i[i_k:k+1] != predict_list_i[i_k:k+1]:
                    FP += 1
                k = g_k + 1
    if TP == FP and TP == FN and TP == 0:
        return 1
    elif (TP == 0 and FN > 0) or (TP == 0 and FP > 0):
        return 0
    else:
        R  = TP / (TP+FN)
        P = TP / (TP+FP)
    return 2*P*R / (P + R)
    pass;
def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx,cx = hidden
    gates = F.linear(input,w_ih,b_ih) + F.linear(hx,w_hh,b_hh)
    _,forgetgate,cellgate,outgate = gates.chunk(4,1)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = forgetgate * cx + (1-forgetgate) * cellgate
    hy = outgate * F.tanh(cy)
    pass;
    return hy,cy
def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    char_embeds = model.char_embeds(batch_char_index_matrices)
    S,W,C,E = char_embeds.size()
    char_embeds = char_embeds.view(S*W,C,E)
    word_len = batch_word_len_lists.view(-1)
    sorted_batch_sentence_len_list,perm_idx  = word_len.sort(0, descending=True)
    sorted_input_embeds = char_embeds[perm_idx]
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_sentence_len_list.data.tolist(), batch_first=True)
    output_sequence, state = model.char_lstm(output_sequence)
    output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
    output_sequence = output_sequence[desorted_indices].view(S*W,C,2,E)
    output = torch.FloatTensor()
    forward = output_sequence[:,:,0,:]
    backward = output_sequence[:,:,1,:]
    for i in range(S*W):
        s_len = word_len[i]
        f= forward[i,s_len-1,:]
        b = backward[i,0,:]
        result = torch.cat([f,b],dim=-1)
        output = torch.cat([output,result],dim=0)
    output = output.view(S,W,2*E)
    return output
    pass;


