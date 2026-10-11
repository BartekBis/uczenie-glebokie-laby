# L1, rzeczy do sprawozdania

**Zad. 4.1 (zad4_1_learning_rate.png):** lr_kryt = 2/λmax = 0.494. Dla 0.98·lr_kryt GD zbiega (23 epoki do 1% od optimum), a dla 1.05·lr_kryt się rozjeżdża, więc hipoteza jest potwierdzona.

**Zad. 6.1 (zad6_1_krzywa_U.png):** Najlepszy stopień wg walidacji to D = 3 (MSE val 0.0911). Błąd train ciągle spada, a walidacyjny od D = 6 rośnie, dlatego hiperparametry wybieram na walidacji, a nie na train.

**Gradienty i kształty:** Poprawność gradientów sprawdzam gradient checkiem (różnica centralna w float64) i porównaniem z autogradem, a kształty printuję przed liczeniem straty.
