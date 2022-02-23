'''
If you want ot see how in practical it would look like: 
https://medium.com/@jinoo/a-simple-example-of-attention-masking-in-transformer-decoder-a6c66757bc7d
https://andrewpeng.dev/content/images/2019/11/encoder-3.png

Code purely taken from:-> https://www.youtube.com/watch?v=U0s0f995w14&t=2103s
'''

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size:int, num_heads:int):
        '''
        https://lilianweng.github.io/lil-log/assets/images/transformer.png
        https://www.youtube.com/watch?v=U0s0f995w14
        '''
        super().__init__()
        assert embed_size // num_heads , "'embed_size' or Embedding Dimension should be perfectly divisible my 'num_heads_parts'"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = self.embed_size // self.num_heads # Embedding is divided into num_heads parts so that each head has same dimension. Dimension of Each Head

        self.key_linear = nn.Linear(self.head_size, self.head_size) # Keys will be multiplied to this weight matrix
        self.query_linear = nn.Linear(self.head_size, self.head_size) # Queries will be multiplied to this weight matrix
        self.value_linear = nn.Linear(self.head_size, self.head_size) # Values will be multiplied to this weight matrix

        self.final_fully_connected = nn.Linear(self.embed_size, self.embed_size) # OR in other terms, nn.Linear(self.num_heads * self.head_size, self.embed_size)
        # Each head's final output will come into this Layer giving a final vector of the size of embedding

    
    def forward(self, key, query, value, mask):
        '''
        For Encoder - Decoder :: Query's next word is dependent on the the words already generated + the attention given to each word in the input sentence
        key: Batch of the sentence we have given as INPUT
        query: Batch of Target Sentences

        key, query and values are batches of sentences of shape [No of samples, Words in each sentence, Embedding Dimension of Each Word]
        '''
        N = key.shape[0] # Number of samples which should be equal in all the key, queries and values
        key_len, query_len, value_len = key.shape[1], query.shape[1], value.shape[1] # number of tokens OR words == length of each sentence 

        assert key_len == value_len, "Key and Value lrngth must be same" # Look at the second einsum where key_len dimension by common "l"

        # Split the Embeddings of key, query and values in [No of samples, Length of each sentence, No of Heads, Dimension of each head]
        query = query.reshape(N, query_len, self.num_heads, self.head_size)
        key = key.reshape(N, key_len, self.num_heads, self.head_size)
        value = value.reshape(N, value_len, self.num_heads, self.head_size)

        key = self.key_linear(key) # These are the weights that will be trained or learned. Aprt from these weights (and the last Linear layer), nothing is trainable
        query = self.query_linear(query)
        value = self.value_linear(value)

        # Look for each block of "Scaled Dot Product Attention" in the image https://lilianweng.github.io/lil-log/assets/images/transformer.png

        # finds out the MATMUL as -> Go each word given in our target, how much ATTENTION do we have to pay for each word in our source
        MATMUL_1 = torch.einsum('nqhd,nkhd->nhqk', [query, key]) # MATMUL has a shape [No of samples, Number of heads, Query length, Key Length]
        # --------- n: Axis for no of samples , q: axis for sentence length in QUERY, k: axis for sentence length in KEY, h: Axis for number of heads, d: Dimension of each head ---------

        SCALE = MATMUL_1 / (self.embed_size ** 0.5) # Scaled Normalization

        if mask is not None: # If mask is given, means we want to cover the value at a particular place inside MATMUL, then keep its value as negative infinity or close to it
            SCALE = SCALE.masked_fill(mask == 0, float('-1e-30')) # When a place inside MASK is 0, means we want to cover it. So fill the particular place inside ENERGY as -infinity
        
        SOFTMAX = torch.softmax(SCALE, dim = 3) # Softmax the scores according to the last axis

        # Now Multiply the Normalized  SOFTMAX to the Value -> Long arrow coming from the beginning in the image given

        MATMUL_2 = torch.einsum('nhql,nlhd->nqhd',[SOFTMAX, value]) # original 'nhqk' is replaced by 'nhql' because k == v == l
        MATMUL_2 = MATMUL_2.reshape(N,query_len,self.embed_size) # embed_size = No of heads * Dimension of each head we need to reshape because we have our Original Embedding fixed

        return self.final_fully_connected(MATMUL_2)


class TransformerBLock(nn.Module): # The part which is repeated Nx times
    '''
    These blocks are repeated "N" times. Each block is having "SelfAttentionBlock", Residual Connections and all
    '''
    def __init__(self, embed_size, num_heads, dropout, forward_expansion:int = 4):
        ''''
        forward_expansion: Expand / Scale the incoming Embedding inside the FEED FORWARD block and then again compress it in the original size
        '''
        super(TransformerBLock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size = embed_size, num_heads = num_heads)

        self.norm1 = nn.LayerNorm(embed_size) # different from BatchNorm. You can "kind of" understand it as "Standardization" vs "Normalization"
        # https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm
        self.norm2 = nn.LayerNorm(embed_size)

        self.FEED_FORWARD = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size , embed_size)
            )
        
        self.dropout_layer = nn.Dropout(dropout)
    

    def forward(self, key , query , value, mask):
        '''
        '''
        attention = self.attention(key = key, query = query, value = value, mask = mask) # Multi Head Attention Block
        skip_connection_1 = self.dropout_layer(self.norm1(attention + query)) # ADD AND NORM Block: Add the original QUERY to the attention : Skip Connection and then apply dropout

        forward = self.FEED_FORWARD(skip_connection_1) # Feed Forward Block. It's output will be added to it's input: Skip Connection
        skip_connection_2 = self.dropout_layer(self.norm2(forward + skip_connection_1)) # ADD and NORM block:  See the Diagram for more clarity
        return skip_connection_2



class Encoder(nn.Module):
    '''
    Encoder Module for Seq - Seq translation type 
    '''
    def __init__(self, embed_size, source_vocab_size, max_len, num_heads, num_trans_blocks, dropout, forward_expansion, device):
        '''
        args:
            max_len: Maximum length of each sentence == Maximum number of tokens / words in each sentence
            num_trans_blocks: No of transformers blocks which will have to be used. This denoted by :Nx: in the diagram
            device: Device type from torch. Be it CUDA or CPU
        '''
        super().__init__()
        self.device = device
        self.embed_size = embed_size
        self.max_len = max_len

        self.word_embeddings = nn.Embedding(source_vocab_size, embed_size) # Embeddings for words. Each word in the vocab will have it's own Embedding Representation
        self.position_embeddings = nn.Embedding(max_len, embed_size) # Learn the Position Embeddings and then add it to the word Embeddings. 1 Embedding per word

        self.transformer_blocks = nn.ModuleList([TransformerBLock(embed_size, num_heads, dropout, forward_expansion) for _ in range(num_trans_blocks)]) # Data will be passed from these many different transformers block in sequential manner. Remember: Each one has it's own weights
        
        # self.transformer_blocks = nn.ModuleList() # Another way of doing this
        # for _ in range(num_trans_blocks): self.transformer_blocks += TransformerBLock(embed_size, num_heads, dropout, forward_expansion)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask):
        '''
        X == Key == Query == Value for Encoder. If you want to visualise, look at the "TRIDENT" shape forming into the Attention block after Embedding. It means one input is repeated 3 times
        x is a batch of sentences where each word is represented by an integer
        '''
        N, max_len = x.shape # [batch size, sentence length]
        positions = torch.arange(0, max_len).expand(N, max_len).to(self.device)  # Shape: No of sentence in batch, Length of each sentence
        # generate N unique number, one for each word even if two words are similar. How to do that? Generate index. : expand is nothing but a loop doing the same for EACH sentence in the batch

        x = self.word_embeddings(x) + self.position_embeddings(positions) # Final embeddings is the sum of these two learned embeddings: Shape: [N, max_len, embed_dim]

        for t_block in self.transformer_blocks:
            x = t_block(key = x, query = x, value = x, mask = mask) # data passes through each block in sequential manner. Since key == query == value so we pass x 3 times
        
        return x


class DecoderBlock(nn.Module): # part which is repeated Nx times
    '''
    If you look closely then the structure is: Masked Multihead Attention -> Add Norm -> @@ Transformer Block @@ -> Linear -> Softmax
    '''
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        '''
        '''
        super().__init__()

        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer = TransformerBLock(embed_size, num_heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, value, key, source_mask, target_mask):
        '''
        x == query
        value == key == Encoder Output
        target_mask is the mask which does not let the model see future predictions
        source_mask is the mask which stops unnecessary calculations which will be caused due to "Padding"
        '''
        attention = self.attention(key = x, query = x, value = x, mask = target_mask) # why 3 x? TRIDENT shape at the starting. Query computes it's self attention
        x = self.dropout(self.norm(attention) + x) # skip connection
        return self.transformer(query = x, value = value, key = key, mask = source_mask) # this mask helps in uncessary gradients computations


class Decoder(nn.Module):
    '''
    DecoderBlock is repeated Nx times
    '''
    def __init__(self, embed_size, target_vocab_size, max_len, num_heads, num_decoder_blocks, dropout, forward_expansion, device):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_size, num_heads, dropout, forward_expansion) for _ in range(num_decoder_blocks)])
        self.output_layer = nn.Linear(embed_size, target_vocab_size) # takes in Embeddings and outputs the probability of each word

    
    def forward(self, x, encoder_state, source_mask, target_mask):
        '''
        x == Query == Input to the decoder
        encoder_state == key == value
        '''
        N, max_len = x.shape

        positions = torch.arange(0,max_len).expand(N,max_len).to(self.device)
        x = self.dropout(self.position_embedding(positions) + self.word_embedding(x))

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_state, encoder_state, source_mask, target_mask) # key == value == encoder_state
        
        return self.output_layer(x)


class Transformer(nn.Module):
    '''
    Encoder-Decoder for Seq-Seq
    '''
    def __init__(self, source_vocab_size, target_vocab_size, embed_size, max_len, source_pad_idx, target_pad_idx, device,
        num_heads = 8, num_trans_blocks = 6, dropout = 0.1, forward_expansion = 4):
        '''
        source_pad_idx: Number which is used to create the Padding for Source Sentence: Will be used to create Source mask
        target_pad_idx: Number which is used to create the Padding for Target Sentence: Will be used to create Target  mask
        '''
        super().__init__()
        self.device = device
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx

        self.encoder = Encoder(embed_size, source_vocab_size, max_len, num_heads, num_trans_blocks, dropout, forward_expansion, device)
        self.decoder = Decoder(embed_size, target_vocab_size, max_len, num_heads, num_trans_blocks, dropout, forward_expansion, device)


    def create_source_mask(self, source):
        '''
        args:
            Input == Source is a 2-D Tensor of [Numbers of samples in a batch, Each Token denoted by an integer]

        Create a same shape tensor according to following rule: 
        1. If it is a padding value, then the resulting position will be 0 else 1
        2. Make it a 4-D Tensor by adding 2 extra dimensions

        Output Dimensions: [N, 1, 1, max_len] :: [Numbers of samples in a batch, Empty axis, Empty axis, 0 or 1 denoting whether to mask this position or not]
        '''
        return (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)


    def create_target_mask(self, target):
        '''
        Input == Target is a 2-D Tensor of [Numbers of samples in a batch, Each Token denoted by an integer]

        Target mask is different from the source mask as (See the image: http://peterbloem.nl/files/transformers/masked-attention.svg):

        "We do not want the decoder to look for words which are in future. For example, if the decoder has already predicted "Hello my name", so we mask the next words
        "is Slim Shady". Doing this, there is no way for the model to cheat and look ahead in future. If we do not mask, model will already know what to predict
        

        process:
        1. Generate a 2-D Tensor of shape [max_len, max_len] which consists of all the ones
        2. Mask ( set values to 0) the Upper Triangular part (area above the Upper left -> Lower right diagonal). the process will be like: 
                For predicting the 3rd word, the model will have access only to the first 2 words nothing more. For 9th word, model will have access to first 8 words
        3. EXPAND or repeat the process the N times (1 mask per sentence in the batch)
        '''
        N, max_len = target.shape
        return torch.tril(torch.ones(max_len,max_len)).expand(N,1,max_len,max_len).to(self.device) # Step 1,2 are done by torch.tril(torch.ones())
        

    def forward(self, source_sentence_batch, target_sentence_batch):
        '''
        '''
        source_mask = self.create_source_mask(source_sentence_batch)
        target_mask = self.create_target_mask(target_sentence_batch)

        encoder_state = self.encoder(source_sentence_batch, source_mask)
        return self.decoder(target_sentence_batch, encoder_state, source_mask, target_mask)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device) # Each number represents a word so 2 samples having 9 words == max_len

    trg = torch.tensor([[1, 7, 4, 11, 5, 9, 2, 0], [1, 5, 6, 2, 13, 7, 6, 2]]).to(device)

    source_pad_idx = 0
    target_pad_idx = 0
    source_vocab_size = 10 # 10 words in source ex: English
    target_vocab_size = 15 # 15 words in target vocab ex: Spanish
    embed_size = 512
    max_len = 100 # Source max len only

    model = Transformer(source_vocab_size, target_vocab_size, embed_size, max_len, source_pad_idx, target_pad_idx, device,
        num_heads = 8, num_trans_blocks = 6, dropout = 0.1, forward_expansion = 4)

    out = model(x, trg[:, :-1])
    print("Success!!!\nOutput Shape: ",out.shape)

































        


