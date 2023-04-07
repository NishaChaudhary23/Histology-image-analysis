#Importing necessary packages
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
paths = ['/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/normal', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/osmf', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/oscc']
datapath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/all'
plotpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/plots/project_1'

if not os.path.exists(datapath):
    os.makedirs(datapath)

master_dataframe = pd.DataFrame()

for i in paths:
    items = os.listdir(i)
    # shuffling the list
    np.random.shuffle(items)
    # taking the first 1000 items
    items = items[:1000]
    print(len(items))
    # adding the path to the items
    items = [os.path.join(datapath, item) for item in items]
    # creating a dataframe
    df = pd.DataFrame(items, columns=['filename'])
    # adding the label
    df['class'] = i.split('/')[-1]
    # appending the dataframe to the master dataframe
    master_dataframe = master_dataframe.append(df)

print(master_dataframe)
# shuffling the master dataframe
# master_dataframe = master_dataframe.sample(frac=1).reset_index(drop=True)
print(master_dataframe)


datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
# Test Data
test_generator = datagen_test.flow_from_dataframe(
        dataframe=master_dataframe,
        folder=datapath,
        target_size=(300, 300),
        class_mode='categorical',
        shuffle=True,
        validate_filenames=False,)


conf_key = [*test_generator.class_indices.keys()]
print(conf_key)

# loading the model
model = load_model('/home/chs.rintu/Documents/office/researchxoscc/Ensemble/models_available/M1/dense169_01.h5')
model.summary()

dense_output = model.get_layer('bn').output
model_dense = Model(inputs=model.input, outputs=dense_output)
model_dense.summary()

# getting the predictions
y_pred = model_dense.predict(test_generator)
print(y_pred)
print(y_pred.shape)
y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1]*y_pred.shape[2]*y_pred.shape[3])

tsne = TSNE(n_components=2, verbose=1)
embeddings = tsne.fit_transform(y_pred)
print(embeddings)

# creating a dataframe
df = pd.DataFrame(embeddings, columns=['x', 'y'])
df['class'] = test_generator.classes
df['class'] = df['class'].map({0:'normal', 1:'oscc', 2:'osmf'})
print(df.head())

# plotting the t-sne plot
plt.figure(figsize=(7, 5))
sns.scatterplot(x='x', y='y', hue='class', data=df, palette=['green', 'orange', 'red'])
# plot titles and other text
plt.title('t-SNE plot of the embeddings from final pooling layer')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
# saving the plot
plt.savefig(os.path.join(plotpath, 'tsne_plot.png'))




