from sklearn.model_selection import train_test_split
import pandas as pd

src_path = './data/train.en'
tgt_path = './data/train.zh'

EOS = 1
SOS = 0


with open(src_path, 'r') as f:
    src_data = f.readlines()
src_data = [i[:-1] for i in src_data if i]


with open(tgt_path, 'r') as f:
    tgt_data = f.readlines()
tgt_data = [i[:-3] for i in tgt_data if i]


x_train, x_test, y_train, y_test = train_test_split(src_data, tgt_data, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)


def save_csv(x, y, save_path):
    corpus = {'English': x, 'Chinese_input': y, 'Chinese_output': y}
    df_corpus = pd.DataFrame(corpus, columns=['English', 'Chinese_input', 'Chinese_output'])

    df_corpus['Chinese_input'] = df_corpus['Chinese_input'].apply(lambda x: str(SOS) + ' ' + x)
    df_corpus['Chinese_output'] = df_corpus['Chinese_output'].apply(lambda x: x + ' ' + str(EOS))


    df_corpus['eng_len'] = df_corpus['English'].apply(lambda x: len(x))
    df_corpus['zh_len'] = df_corpus['Chinese_input'].apply(lambda x: len(x))

    df_corpus = df_corpus.query('eng_len < 80 & zh_len < 80')
    df_corpus = df_corpus.query('eng_len > 5 & zh_len > 5')

    df_corpus.to_csv(save_path, columns=['English', 'Chinese_input', 'Chinese_output'], index=False)


save_csv(x_train, y_train, './data/train.csv')
save_csv(x_test, y_test, './data/test.csv')
save_csv(x_val, y_val, './data/dev.csv')
