import torch
from config import config
_config = config()
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
def evaluate(golden_list, predict_list):
    pass;
    # initialize FN, FP, TP
    # golden_list =  [['B-TAR', 'B-TAR', 'O', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
    # predict_list = [['B-TAR', 'B-TAR', 'O', 'O'], ['B-TAR', 'O', 'B-HYP', 'I-HYP']]
    FN,FP,TP = 0,0,0
    external_len = len(golden_list)
    # identify FN and TP
    for k in range(external_len):
        g_list = golden_list[k]
        p_list = predict_list[k]
        g_list_len = len(g_list)
        i = 0
        while (i<g_list_len):
            symbol = g_list[i]
            if symbol == 'O':
                i += 1
                continue
            else:
                symbol_FN = symbol[2:]
                d = i
                if d < g_list_len-1:
                    while (g_list[d+1][2:] == symbol_FN or p_list[d+1][2:]== symbol_FN):
                        d += 1
                        if d == g_list_len-1:
                            break
                if g_list[i:d+1] != p_list[i:d+1]:
                    FN += 1
                else:
                    TP += 1
            if g_list[d][2:] == symbol_FN:
                i = d+1
            else:
                i = i+1
    # identify FP
        i = 0
        while (i<g_list_len):
            symbol = p_list[i]
            if symbol == 'O':
                i += 1
                continue
            else:
                symbol_FP = symbol[2:]
                d = i
                if d < g_list_len -1:
                    while (p_list[d+1][2:] == symbol_FP or g_list[d+1][2:] == symbol_FP):
                        d += 1
                        if d == g_list_len-1:
                            break
                if g_list[i:d+1] != p_list[i:d+1]:
                    FP += 1
            if p_list[d][2:] == symbol_FP:
                i = d+1
            else:
                i += 1
    Precion = TP / (TP+FP)
    Recall  = TP / (TP+FN)
    if (Precion + Recall) == 0:
        return 0
    else:
        return 2*Precion*Recall / (Precion + Recall)
    
def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx,cx = hidden
    gates = F.linear(input,w_ih,b_ih) + F.linear(hx,w_hh,b_hh)
    ingate, forgetgate,cellgate,outgate = gates.chunk(4,1)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = forgetgate * cx + (1-forgetgate) * cellgate
    hy = outgate * F.tanh(cy)
    pass;
    return hy,cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    pass;
    final_result = torch.FloatTensor()

    #for each sentence
    for sent in range(len(batch_char_index_matrices)):
        
        #char_embed
        input_char_embeds = model.char_embeds(batch_char_index_matrices[sent])
        
        #sort
        perm_idx, sorted_batch_word_len_list = model.sort_input(batch_word_len_lists[sent])
        sorted_input_embeds = input_char_embeds[perm_idx]
        _, desorted_indices = torch.sort(perm_idx, descending=False)

        #char_lstm
        output_sequence = torch.nn.utils.rnn.pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_word_len_list.data.tolist(), batch_first=True)
        output_sequence, _ = model.char_lstm(output_sequence)
        output_sequence, _ = torch.nn.utils.rnn.pad_packed_sequence(output_sequence, batch_first=True)
        output_sequence = output_sequence[desorted_indices]

        #last vector of forward result
        forward_last = torch.FloatTensor()
        for i, lenth in enumerate(batch_word_len_lists[sent]):
            forward_last = torch.cat([forward_last, output_sequence[i,int(lenth)-1,:50][None,:]],0)
        
        #first vector of backward result
        backward_first = output_sequence[:, 0, 50:]
        
        #append two vectors
        result = torch.cat([forward_last,backward_first],-1)

        final_result = torch.cat([final_result, result[None,:]], 0)

    return final_result


