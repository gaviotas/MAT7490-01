from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import pandas as pd

path_gothic = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
prop = fm.FontProperties(fname=path_gothic)
matplotlib.rcParams["axes.unicode_minus"] = False


def generate_plot(vocab, X_w2v):
    tsne = TSNE(n_components=2)
    X_w2v_tsne = tsne.fit_transform(X_w2v)

    df = pd.DataFrame(X_w2v_tsne, index=vocab, columns=["x", "y"])
    # %matplotlib inline

    fig = plt.figure()
    fig.set_size_inches(40, 20)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df["x"], df["y"])

    for word, pos in list(df.iterrows()):
        ax.annotate(word, pos, fontsize=12, fontproperties=prop)
    plt.show()
