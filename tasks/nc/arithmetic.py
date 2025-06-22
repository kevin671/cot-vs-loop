
class Arithmetic(task.):
    def toToken(sentences):
        token_list = list()
        for sentence in sentences:
            arr = [dictionary[s] for s in sentence.split()] + [2]
            padding = [0 for _ in range(args.maxdata - len(arr))]
            arr = arr + padding
            token_list.append(torch.Tensor(arr))
        return torch.stack(token_list).int()

    def getY(X, chain):
        if not chain:
            x = torch.where(X==dictionary['='], 1, 0)
            Y = X[:, 1:] * x[:, :-1]
        else:
            Y = X[:, 1:] * 1
            b = Y.shape[0]
            equa = torch.argmax(torch.where(Y==dictionary['='], 1, 0), dim=1)
            eos = torch.argmax(torch.where(Y==dictionary['<eos>'], 1, 0), dim=1)
            for i in range(b):
                Y[i, :equa[i] + 1] = 0
                Y[i, eos[i]+1:] = 0
        return Y

def generateArithmeticData(args, dictionary):
    import random
    import torch
    import task

    def generate_sentence():
        first_number = random.randint(0, 2**args.maxdata - 1)
        second_number = random.randint(0, 2**args.maxdata - 1)
        result = first_number + second_number
        sentence = f"{first_number:0{args.maxdata}b}+{second_number:0{args.maxdata}b}={result:0{args.maxdata}b}<eos>"
        return sentence

    sentences = [generate_sentence() for _ in range(args.num_samples)]
    return Arithmetic.toToken(sentences)