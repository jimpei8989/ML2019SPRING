import numpy as np
import matplotlib.pyplot as plt

lamb = np.array([0.1, 0.01, 0.001, 0.0001])
Epub = np.array([5.77185, 5.77391, 5.77412, 5.77414])
Epri = np.array([7.27943, 7.27933, 7.27932, 7.27932])

fig, (pubplt, priplt) = plt.subplots(1, 2)

plt.subplots_adjust(wspace = 0.8)
pubplt.set_title("Public Score")
pubplt.semilogx(lamb, Epub, color = 'aqua')
pubplt.set_xlabel("$\lambda$")
pubplt.set_ylabel("Score")

priplt.set_title("Private Score")
priplt.semilogx(lamb, Epri, color = 'magenta')
priplt.set_xlabel("$\lambda$")
priplt.set_ylabel("Score")
:qa

plt.savefig("plot_kaggle.png", dpi = 400)
