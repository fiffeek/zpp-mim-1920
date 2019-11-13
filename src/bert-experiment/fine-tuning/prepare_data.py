import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('bert/fine-tune_data/auto-tune.csv')

enc = LabelEncoder()
df.columns = ['id', 'tactic', 'params', 'theorem']
df = df[df['theorem'].str.len().lt(1500)]
df['tactic'] = enc.fit_transform(df['tactic'])
df['id'] = ['id' + str(i)  for i in range(df.shape[0])] 
#TODO maybe change the classifier to find list of tactics?
df['params'] = ['a'] * df.shape[0]

df = df[df['theorem'].str.len().lt(1500)]

df_train, df_test_dev = train_test_split(df, test_size=0.2)
df_test, df_dev = train_test_split(df_test_dev, test_size = 0.5)

df_bert_train = pd.DataFrame({'guid': df_train['id'],
    'label': df_train['tactic'],
    'alpha': df_train['params'],
    'text': df_train['theorem']})

df_bert_dev = pd.DataFrame({'guid': df_dev['id'],
    'label': df_dev['tactic'],
    'alpha': df_dev['params'],
    'text': df_dev['theorem']})

df_bert_test = pd.DataFrame({'guid': df_test['id'],
    'text': df_test['theorem']})

df_bert_train.to_csv('bert/fine-tune_data/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('bert/fine-tune_data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('bert/fine-tune_data/test.tsv', sep='\t', index=False, header=True)
