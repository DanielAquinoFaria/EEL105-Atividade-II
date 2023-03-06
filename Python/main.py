import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

comprimento = 0.02
largura = 0.010
espessura = 0.001
corrente = 0.005
boltzman = 8.6173324 * pow(10, -5)

areaTransversal = largura * espessura
alpha = (comprimento * corrente) / areaTransversal
print(alpha)

data = pd.read_csv('dados.csv')  # C, Vp, K
dataKinv = pow(data['K'], -1)  # inverso da temperatura em kelvin

condutividade = alpha / data['Vp']
logCondutividade = np.log(condutividade)

# grafico 1 condutividade por temperatura
f = plt.figure(1)
plt.title("Condutividade x Temperatura")
plt.xlabel(r"$\mathrm{{T}\left[{K}\right]}$", fontsize="12")
plt.ylabel(r'$\mathrm{\sigma\left[\frac{1}{\Omega.m}\right]}$', rotation=90, fontsize="13")
plt.grid(linestyle='--')
plt.scatter(data['K'], condutividade)

tempdataKin = np.vstack(dataKinv)  # agrupando os valores de dataKinv em um vetor 2d
templogCondutividade = np.vstack(logCondutividade)  # agrupando os valores de condutividade em um vetor 2d
reg = linear_model.LinearRegression()  # regressão linear
reg.fit(tempdataKin, templogCondutividade)
regX = np.linspace(0.0022, 0.0034, 100).reshape(-1, 1)  # variaveis para a função de onda do erro
regY = reg.predict(regX)
score = reg.score(tempdataKin, templogCondutividade)  # R²

g = plt.figure(2)
plt.title("Condutividade x Inverso da Temperatura")
plt.xlabel(r"$\mathrm{\frac{1}{T}\left[\frac{1}{K}\right]}$", fontsize="15")
plt.ylabel(r'$\mathrm{\ln{\sigma}\left[\frac{1}{\Omega.m}\right]}$', rotation=90, fontsize="13")
plt.grid(linestyle='--')
plt.scatter(dataKinv, logCondutividade)
legenda = r"$\mathrm{\sigma = %0.2f*T + %0.2f}$" "\n" r"$\mathrm{R^2 = %0.2f}$" % (reg.coef_, reg.intercept_, score)
plt.plot(regX, regY, color="r", label=legenda)
plt.legend()
plt.tight_layout()
plt.show(block=False)

eg = - (reg.coef_ * (2 * boltzman))

print('Eh = %0.2f eV' % eg)

plt.show()
