import torch
import torch.nn as nn
from queue import PriorityQueue
import operator

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(decoder, batch_size, decoder_hidden, device, encoder_output=None, context = None):
    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []
    decoder_hidden = None

    # decoding goes sentence by sentence
    for idx in range(batch_size):

        try:
            # print(idx)
            # Start with the start of the sentence token
            decoder_input = torch.LongTensor(context, device=device)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                if qsize > 2000: break

                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid[0].item() == context[0] and n.prevNode != None:
                    endnodes.append((score, n))
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

                log_prob, indexes = torch.topk(decoder_output, beam_width)
                if len(log_prob.size()) == 3:
                    log_prob = log_prob[0]
                    indexes = indexes[0]
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                qsize += len(nextnodes) - 1


            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid[0].item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid[0].item())

                utterance = utterance[::-1]
                utterances.append(utterance)

            # print(utterances)
            # print("--------------")

            decoded_batch.append(torch.Tensor(utterances[0]))
        except:
            return decoded_batch

    return decoded_batch


