from char_map import index_map, char_map
import numpy as np
from nltk.metrics import distance
import itertools
# index_map = {0: ' ', 1: "'", 2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 
#              9: 'h', 10: 'i', 11: 'j', 12: 'k', 13: 'l', 14: 'm', 15: 'n', 16: 'o', 
#              17: 'p', 18: 'q', 19: 'r', 20: 's', 21: 't', 22: 'u', 23: 'v', 24: 'w', 
#              25: 'x', 26: 'y', 27: 'z'}
# trong mô hình thì index 28 chính là kí tự đặc biệt blank, ta sẽ xóa đi tất cả các kí tự này
# còn các kí tự còn lại thì sẽ gộp lại theo giải thuật CTC, những kí tự gần nhau
# giống nhau sẽ được gộp lại thành một kí tự duy nhất
# đánh giá mô hình dựa trên thang đo word error rate
BLANK_CHAR_INDEX = 94
def sequence_index_to_string(sequence) :
    """
        biến đổi chuỗi các số thành chuỗi các kí tự tương ứng
        các số trong sequence nên nằm trong phạm vi từ 0 - 27
    """
    leter_sequence = []

    for i in range(np.size(sequence)) :
        leter_sequence.append(index_map[sequence[i]])
    
    return leter_sequence

def sequence_index_to_string_test() :
    sequence = [2, 3, 4, 5, 6, 7]
    result = ['a', 'b', 'c', 'd', 'e', 'f']
    # print(sequence_index_to_string(sequence))
    assert(sequence_index_to_string(sequence) == result)
    print("test sequence_index_to_string success")


def single_instance_decode(logit, length) :
    """
        logit là một array có shape là [timesteps, num_classes]
        length là độ dài thực tế của timesteps dùng để dự đoán chuỗi kí tự

        output là chuỗn các kí tự được tính bằng cách lấy giá trị lớn nhất 
        tại mỗi timestep của logit
    """

    list_char = list(np.argmax(logit[0:length], axis=1))
    
    list_char = [k for k, _ in itertools.groupby(list_char)]
    
    result = []
    for k in list_char:
      if k != BLANK_CHAR_INDEX:
        result.append(k)

    return sequence_index_to_string(result)

def batch_decode(logits, sequence_lengths) :
    """
        logits là output của model, có shape là [batch_size, timestep, num_classes]
        sequence_lengths là array có shape [batch_size],
        sequence_lengths dùng để lưu lại số timesteps thực chất

        output là một array có size là batch_size với mỗi phần tử là một list của các
        kí tự mà mô hình dự đoán được
    """
    result = []

    for i in range(np.size(sequence_lengths)) :
        result.append(single_instance_decode(logits[i], sequence_lengths[i]))

    return result

def single_label_to_text(label, length) :
    """
        dùng hàm này để chuyển một nhãn của câu sang kí tự
    """
    return sequence_index_to_string(label[0:length])

def batch_label_to_text(labels, label_lengths) :
    """
        input là một batch của labels và độ dài tương ứng với label đó

        output là list của các label được chuyển từ index sang kí tự
    """
    result = []
    for i in range(np.size(label_lengths)) :
        result.append(single_label_to_text(labels[i], label_lengths[i]))
    
    return result

def list_char_to_string(sequence) :
    """
        hàm này dùng để chuyển từ list các character
        sang chuỗi string cho dễ quan sát
    """
    return ''.join(sequence)

def list_char_to_string_test() :
    sequence = ['a', 'b', 'c', 'd']
    assert(list_char_to_string(sequence) == 'abcd')
    print("test list_char_to_string pass")


# cách tính cer và wer tham khảo từ chương trình deep speech 2 của tensorflow
def compute_cer(predict, label) :
    """
        Dùng để tính character error rate
        predict :   string dự đoán được từ mô hình
        label   :   string true label

        chương trình trả về character error rate
    """
    return distance.edit_distance(predict, label)/len(label)

def compute_wer(predict, label) :
    """
        Dùng để tính character error rate
        predict :   string dự đoán được từ mô hình
        label   :   string true label

        chương trình trả về word error rate
    """
    words = set(predict.split() + label.split())
    word2char = dict(zip(words, range(len(words))))

    new_predict = [chr(word2char[w]) for w in predict.split()]
    new_label = [chr(word2char[w]) for w in label.split()]
    label_size = np.size(new_label)

    return distance.edit_distance(''.join(new_predict), ''.join(new_label))/label_size
