import codecs
import os
import collections
from six.moves import cPickle

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding="utf-8"):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.txt")
        tensor_file = os.path.join(data_dir, "data.npy")

        # vocab, tensor file 없으면 전처리 시작
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
            self.create_batches() # 배치 생성
            self.reset_batch_pointer() # 배치 포인터
    
    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()

            # 문자(character) 별 등장횟수를 센다.
            counter = collections.Counter(data)
            count_pairs = sorted(counter.items(), key=lambda x: -x[1])
            self.chars, _ = zip(*count_pairs) # 전체 문자들
            self.vocab_size = len(self.chars) # 전체 문자 개수
            self.vocab = dict(zip(self.chars, range(len(self.chars))))

            # 단어들을 (char, id) 형태의 dictionary 로 만든다
            with open(vocab_file, "wb") as f:
                cPickle.dump(self.chars, f)

            # 데이터의 각각의 char들을 id로 변경
            self.tensor = np.array(list(map(self.vocab.get, data)))
            np.save(tensor_file, self.tensor)
    
    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
            self.vocab_size = len(self.chars)
            self.vocab = dict(zip(self.chars, range(len(self.chars))))
            self.tensor = np.load(tensor_file)
            self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
            
