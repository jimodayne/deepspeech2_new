import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

def stretch_data(data, rate) :
    return librosa.effects.time_stretch(data, rate)

def speed_augmentation(data) :
    rand_num = random.uniform(0.8, 1.1)
    # print(rand_num)
    result = stretch_data(data, rand_num)
    return result

def change_volume(data) :
    max_ampli = np.max(data)
    threshold = 1 / max_ampli
    volume_ratio = random.uniform(0.1, threshold)
    change_volume_data = np.asarray(data) * volume_ratio
    # print(volume_ratio)
    # print(threshold)
    return change_volume_data

def augmentation_clean_data(data) :
    # thuc hien change speed truoc sau do change volume
    change_speed_data = speed_augmentation(data)
    return change_volume(change_speed_data) 


def compute_db(data) :
    average = np.sum(np.absolute(data)) / len(data)
    result = 10 * np.log10(average)
    return result 

def add_background_noise(audio_data, noise_data) :

    my_data = np.asarray(audio_data)

    audio_data_db = compute_db(audio_data)
    noise_data_db = compute_db(noise_data)
    ratio = noise_data_db / (audio_data_db + noise_data_db)
    random_num = random.uniform(0, ratio * 2 / 3)
    # print(random_num)
    my_data = np.multiply((1 - random_num) , my_data) + np.multiply(random_num, noise_data[0: len(my_data)])
        
    return my_data

def augmentation_background_noise(data, noise_data) :
    change_speed_data = speed_augmentation(data)
    change_volume_data = change_volume(change_speed_data) 
    return add_background_noise(change_volume_data, noise_data)

def test_augmentation_clean_data(input_file, output_file, fs=44100) :

    data, fs = librosa.load(input_file, fs)

    plt.figure(0)
    plt.title('original data')
    plt.plot(data)
    plt.legend(loc='upper right')
    
    augmentation_data = augmentation_clean_data(data)
    
    plt.figure(1)
    plt.title('augmentation data')
    plt.plot(augmentation_data)
    plt.legend(loc='upper right')
    
    librosa.output.write_wav(output_file, augmentation_data, 44100)

    plt.show()

def test_augmentation_background_noise_data(input_file, noise_file, output_file, fs = 44100) :

    noise_data, noise_fs = librosa.load(noise_file, fs)

    noise_random_range = len(noise_data) - 10 * noise_fs
    rand_noise_pos = random.randint(0, noise_random_range)

    data, fs = librosa.load(input_file, fs)
    plt.figure(0)
    plt.title('original data')
    plt.plot(data)
    plt.legend(loc='upper right')
    
    augmentation_data = augmentation_background_noise(data, noise_data[rand_noise_pos : ])
    
    plt.figure(1)
    plt.title('augmentation data')
    plt.plot(augmentation_data)
    plt.legend(loc='upper right')
    
    librosa.output.write_wav(output_file, augmentation_data, 44100)

    plt.show()