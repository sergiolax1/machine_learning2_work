from fastapi import FastAPI
from tensorflow.keras.datasets import mnist
from unsupervised import PCA_U, TSNE_U
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split

app = FastAPI()

@app.get("/point11")
async def prediction():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_images = train_images.reshape(-1, 28 * 28)
    test_images = test_images.reshape(-1, 28 * 28)

    train_filter = np.where((train_labels == 0) | (train_labels == 8))
    test_filter = np.where((test_labels == 0) | (test_labels == 8))

    train_images, train_labels = train_images[train_filter], train_labels[train_filter]
    test_images, test_labels = test_images[test_filter], test_labels[test_filter]

    X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.3, random_state=256)
    pca = PCA_U(n_components=2)
    pca.fit(X_train)
    x_train_pca = pca.transform(X_train)
    clf_pca = LogisticRegression(random_state=345)
    clf_pca.fit(x_train_pca, y_train)
    n_sample = 2
    random_sample_idx = np.random.choice(X_train.shape[0], n_sample, replace=False)
    sample_pca = x_train_pca[random_sample_idx, :]
    prediction_pca = clf_pca.predict(sample_pca)
    return {'prediccion pca': prediction_pca.tolist()}

### se evalua con uvicorn point11:app --reload"
