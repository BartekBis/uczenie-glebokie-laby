# L2, rzeczy do sprawozdania

**Liczba parametrów (zad. 4.1):** Warstwa gęsta ma (n_in + 1)·n_out parametrów. Dla [2, 16, 16, 1] wychodzi 48 + 272 + 17 = 337, a dla [2, 64, 32, 8, 1] wychodzi 192 + 2080 + 264 + 9 = 2545, w obu przypadkach tak samo jak `count_params()` w Keras. Tak samo policzę architekturę z mini projektu.

**Zad. 6.1 (zad6_1_siatka_architektur.png):** Na make_moons z szumem 0.3 i 200 próbkami najlepsze na walidacji jest 1 warstwa × 8 neuronów (val 0.91). Model 1 × 2 jest niedouczony (0.85 i 0.85), a 2 × 64 i 3 × 64 są przeuczone (train 0.97 i 1.00, val 0.89). Rozmiar sieci wybieram więc na walidacji.

**Zad. 5.1:** `trenuj_epoke` i `ewaluuj` są przetestowane: na make_moons dały val acc 0.988, czyli tyle co Keras (0.980). Można ich użyć w mini projekcie, jeśli będzie w PyTorch.

**Zad. 2.2 (zad2_2_znikajacy_gradient.png):** W 20 warstwach z sigmoidem gradient w warstwie 1 jest 13 rzędów wielkości mniejszy niż w ostatniej, a z ReLU i He zostaje w tym samym rzędzie.
