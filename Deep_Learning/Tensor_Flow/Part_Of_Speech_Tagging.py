import sys
import numpy
import re
import tensorflow as tf

TRAIN_TIME_MINUTES = 6


class DatasetReader(object):

    @staticmethod
    def index_value(term_index, tag_index):
        "In order to ensure that all index values from 0 to number of unique terms-1 is utilized for both terms and tags, I am ensuring that I take a variable as index value holder         and increment its value cautiously when an unique term is observed"
        if len(term_index) == 0:
            term_index_value = 0
        else:
            term_index_value = max(term_index.values()) + 1
        
        if len(tag_index) == 0:
            tag_index_value = 0
        else:
            tag_index_value = max(tag_index.values()) + 1
        
        return (term_index_value,tag_index_value)
    
    
    @staticmethod
    def Read_File(filename):
        "The content from files are read and its contents are observed and split on rightmost / so that we can obtain the term and the tag element and create term_index and          tag_index dictionary accordingly"
        with open(filename, 'r+') as finput:
            comment = finput.read()
            training_lines = comment.splitlines()
        finput.close()
        return training_lines
        
        
    
    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        training_data = []
        """Reads file into dataset, while populating term_index and tag_index.
     
        Args:
            filename: Path of text file containing sentences and tags. Each line is a
                sentence and each term is followed by "/tag". Note: some terms might
                have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
                separates the tag.
            term_index: dictionary to be populated with every unique term (i.e. before
                the last "/") to point to an integer. All integers must be utilized from
                0 to number of unique terms - 1, without any gaps nor repetitions.
            tag_index: same as term_index, but for tags.

        the _index dictionaries are guaranteed to have no gaps when the method is
        called i.e. all integers in [0, len(*_index)-1] will be used as values.
        You must preserve the no-gaps property!

        Return:
            The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
            each parsedLine is a list: [(termId1, tagId1), (termId2, tagId2), ...] 
        """
         
        index_value=DatasetReader.index_value(term_index, tag_index)
        term_index_value=index_value[0]
        tag_index_value=index_value[1]
        training_lines=DatasetReader.Read_File(filename)
        
        
        
        for itemx in training_lines:
            line_list = []
            for item in re.split(' ', itemx) :
                element = item.rsplit('/', 1) 
                
                
                
                
                
                if element[0] in term_index.keys():
                    item1 = term_index[element[0]]
                else:
                    term_index[element[0]] = int(term_index_value)
                    item1 = term_index_value
                    term_index_value = term_index_value + 1
                
     
                if element[1] in tag_index.keys():
                    item2 = tag_index[element[1]]
                else:
                    tag_index[element[1]] = int(tag_index_value)
                    item2 = tag_index_value
                    tag_index_value = tag_index_value + 1
                 
                
                
                
                
                
                line_list.append((item1, item2))
            training_data.append(line_list)       
        return training_data


    @staticmethod
    def BuildMatrices(dataset):
        term_matrix=[]
        tag_matrix=[]
        """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

        Args:
            dataset: Returned by method ReadFile. It is a list (length N) of lists:
                [sentence1, sentence2, ...], where every sentence is a list:
                [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

        Returns:
            Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
                terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
                    indices in dataset[i].
                tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
                    indices in dataset[i].
                lengths: shape (N) int64 numpy array. Entry i contains the length of
                    sentence in dataset[i].

            T is the maximum length. For example, calling as:
                BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
            i.e. with two sentences, first with length 2 and second with length 4,
            should return the tuple:
            (
                [[1, 4, 0, 0],    # Note: 0 padding.
                 [13, 3, 7, 3]],

                [[2, 10, 0, 0],   # Note: 0 padding.
                 [20, 6, 8, 20]], 

                [2, 4]
            )
        """
        length_matrix=[len(i) for i in numpy.array(dataset)]
        
       
        for line in dataset:
            term_matrix.append([x[0] for x in line]+ [0]*(max(length_matrix)-len(line)))
            tag_matrix.append([x[1] for x in line]+ [0]*(max(length_matrix)-len(line)))
        
        
        return numpy.asarray(term_matrix), numpy.asarray(tag_matrix), numpy.asarray(length_matrix)
   

    @staticmethod
    def ReadData(train_filename, test_filename=None):
        """Returns numpy arrays and indices for train (and optionally test) data.
        
        Args:
            train_filename: .txt path containing training data, one line per sentence.
                The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
            test_filename: Optional .txt path containing test data.

        Returns:
            A tuple of 3-elements or 4-elements, the later iff test_filename is given.
            The first 2 elements are term_index and tag_index, which are dictionaries,
            respectively, from term to integer ID and from tag to integer ID. The int
            IDs are used in the numpy matrices.
            The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
                - train_terms: numpy int matrix.
                - train_tags: numpy int matrix.
                - train_lengths: numpy int vector.
                These 3 are identical to what is returned by BuildMatrices().
            The 4th element is a tuple of 3 elements as above, but the data is
            extracted from test_filename.
        """
        term_index = {'_oov_': 0}  # Out-of-vocab is term 0.
        tag_index = {}

        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)

        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                    (train_terms, train_tags, train_lengths),
                    (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)


        
class SequenceModel(object):

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        """Constructor. You can add code but do not remove any code.

        The arguments are arbitrary: when you are training on your own, PLEASE set
        them to the correct values (e.g. from main()).

        Args:
            max_lengths: maximum possible sentence length.
            num_terms: the vocabulary size (number of terms).
            num_tags: the size of the output space (number of tags).

        """
        self.maximum_length = max_length
        self.number_of_terms = num_terms
        self.number_of_tags = num_tags
        self.slice_x = tf.placeholder(tf.int64, [None, self.maximum_length], 'X')
        self.lengths = tf.placeholder(tf.int32, [None], 'lengths')
        self.slice_y = tf.placeholder(tf.int64,[None, self.maximum_length] , 'Y')
        self.sess1=tf.Session()

    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.
        
        Specifically, the return matrix B will have the following:
            B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        return tf.sequence_mask(length_vector, maxlen=self.maximum_length, dtype=tf.float32, name=None)

    def save_model(self, filename):
        """Saves model to a file."""
        pass

    def load_model(self, filename):
        """Loads model from a file."""
        pass

    def build_inference(self):
        """Build the expression from (self.x, self.lengths) to (self.logits).
        
        Please do not change or override self.x nor self.lengths in this function.

        Hint:
            - Use lengths_vector_to_binary_matrix
            - You might use tf.reshape, tf.cast, and/or tensor broadcasting.
        """
        state_size = self.number_of_tags
        embed_matrix = tf.get_variable('embeddings', [self.number_of_terms, self.number_of_tags])
        xemb = tf.nn.embedding_lookup(embed_matrix, self.slice_x)
        fw = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.0)
        bw = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.0)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw, bw, xemb, sequence_length=self.lengths, dtype=tf.float32)
        layer = tf.concat(outputs, -1)
        self.logits = tf.layers.dense(layer, state_size)

    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths.
        
        Hint: This function is straight forward and you might find this code useful:
        # logits = session.run(self.logits, {self.x: terms, self.lengths: lengths})
        # return numpy.argmax(logits, axis=2)

        Args:
            terms: numpy int matrix, like terms_matrix made by BuildMatrices.
            lengths: numpy int vector, like lengths made by BuildMatrices.

        Returns:
            numpy int matrix of the predicted tags, with shape identical to the int
            matrix tags i.e. each term must have its associated tag. The caller will
            *not* process the output tags beyond the sentence length i.e. you can have
            arbitrary values beyond length.
        """
        logits = self.sess1.run(self.logits, {self.slice_x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)

    def build_training(self):
        """Prepares the class for training.
        
        It is up to you how you implement this function, as long as train_on_batch
        works.
        
        Hint:
            - Lookup tf.contrib.seq2seq.sequence_loss 
            - tf.losses.get_total_loss() should return a valid tensor (without raising
                an exception). Equivalently, tf.losses.get_losses() should return a
                non-empty list.
        """
        learning_rate1=1e-2
        loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(self.logits, self.slice_y, self.lengths_vector_to_binary_matrix(self.lengths)))
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate1).minimize(loss)
        self.sess1.run(tf.global_variables_initializer())
        
        
    def train_epoch(self, terms, tags, lengths, batch_size=40, learn_rate=1e-2):
        """Performs updates on the model given training training data.
        
        This will be called with numpy arrays similar to the ones created in 
        Args:
            terms: int64 numpy array of size (# sentences, max sentence length)
            tags: int64 numpy array of size (# sentences, max sentence length)
            lengths:
            batch_size: int indicating batch size. Grader script will not pass this,
                but it is only here so that you can experiment with a "good batch size"
                from your main block.
            learn_rate: float for learning rate. Grader script will not pass this,
                but it is only here so that you can experiment with a "good learn rate"
                from your main block.

        Return:
            boolean. You should return True iff you want the training to continue. If
            you return False (or do not return anyhting) then training will stop after
            the first iteration!
        """
        # Finally, make sure you uncomment the `return True` below.
        indices = numpy.random.permutation(terms.shape[0])
        for item in range(0, terms.shape[0], batch_size):
            element = min(item + batch_size, terms.shape[0])
            slice_x1 = terms[indices[item:element]] + 0
            slice_y1 = tags[indices[item:element]] + 0
            slice_lengths = lengths[indices[item:element]] + 0
            self.sess1.run(self.opt, {self.slice_x:slice_x1, self.slice_y:slice_y1, self.lengths:slice_lengths}) 
        return True
        

    def evaluate(self, terms, tags, lengths):
        pass


def main():
    """This will never be called by us, but you are encouraged to implement it for
    local debugging e.g. to get a good model and good hyper-parameters (learning
    rate, batch size, etc)."""
    # Read dataset.
    reader = DatasetReader
    train_filename = sys.argv[1]
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    for j in xrange(10):
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (j+1))
        model.evaluate(test_terms, test_tags, test_lengths)


if __name__ == '__main__':
    main()
