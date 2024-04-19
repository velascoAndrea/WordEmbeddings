#!/usr/bin/env python
# coding: utf-8

# In[34]:


with open("molvufile.txt", "r") as file:
    content = file.read()
print(content)


# In[35]:


def remove_newlines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    serie = serie.replace('  ', ' ')
    return serie

remove_newlines(content)


# In[36]:


import pandas as pd

df = pd.DataFrame({'text': [remove_newlines(content)]})
df.to_csv('processed.csv', index=False)
df.head()


# In[37]:


import tiktoken
import matplotlib
tokenizer = tiktoken.get_encoding("cl100k_base")
df = pd.read_csv('processed.csv')
df.columns = ['text']
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()


# In[38]:


max_tokens = 150
# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):
    # Split the text into sentences
    sentences = text.split('.\n')
    
    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    chunks = []
    tokens_so_far = 0
    chunk = []
    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0
        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue
        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1
    return chunks
# shortened = []

# shortened = split_into_many(content)
shortened = []
# print(shortened)
# print(len(shortened))
for row in df.iterrows():
    if row[1]['text'] is None:
        continue
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(content)
    else:
        shortened.append(content)


# In[39]:


df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()


# In[41]:


import numpy as np
from scipy.spatial.distance import cosine

df=pd.read_csv('embedings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()



# In[42]:


def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    q_embeddings = get_embedding(question)
    # q_embeddings = client.embeddings.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    # Get the distances from the embeddings
    df["distances"] = df["embeddings"].apply(lambda x: cosine(q_embeddings, x))

    returns = []
    cur_len = 0
    
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        cur_len += row['n_tokens'] + 4
        if cur_len > max_len:
            break
        returns.append(str(i))
    return ", ".join(returns)


# In[43]:


def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=500,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Contexto:\n")
        print(context)
        print("\n\n")

    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Responde la pregunta basandote en el contexto, hazlo de manera clara. Si no puedes reponder basado en el contexto, solamente di 'Escribe AGENTE para más información.'"},
                {"role": "system", "content": f"contexto: {context}"},
                {"role": "user", "content": f"pregunta: {question}"},
                {"role": "assistant", "content": "respuesta:"}
                # {"role": "user", f"content": "Contexto: {context}\n\n---\n\nPregunta: {question}\nRespuesta:"}
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""


# In[32]:


answer_question(df, question="¿Que mochila tiene el espacio mas grande para almacenar mi laptop?", max_tokens=100)


# In[12]:


answer_question(df, question="¿Cual es la mochila mas amplia?")


# In[13]:


answer_question(df, question="Me gusta acampar ¿Qué mochila me recomiendas?", max_tokens=100)


# In[14]:


answer_question(df, question="¿Qué mochila es la más liviana?", max_tokens=100)


# In[15]:


answer_question(df, question="¿Qué mochila es la mas pequeña y portatil?", max_tokens=100)


# In[44]:


answer_question(df, question="¿Cual es la mejor mochila?", max_tokens=100)


# In[ ]:




