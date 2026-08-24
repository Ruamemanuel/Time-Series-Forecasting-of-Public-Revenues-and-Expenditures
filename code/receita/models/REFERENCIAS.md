# Justificativa dos grids de hiperparâmetros (modelos locais)

Referências que embasam a escolha dos grids de SVR e MLP em `local_model.py`.
Mantido fora do código para deixar o módulo enxuto; não precisa ir para o repositório.

## SVR (24 combinações, kernel RBF)

- **Kernel RBF apenas** — Kim (2003) mostra que o RBF supera linear e polinomial
  em séries financeiras mensais; Tay & Cao (2001) chegam à mesma conclusão. O
  linear é descartado porque as séries fiscais têm não-linearidades relevantes
  (sazonalidade, quebras estruturais).
- **C = {0.01, 0.1, 1, 10, 100, 1000}** — grade exponencial base 10 recomendada
  por Hsu, Chang & Lin (2016). Cobre quatro ordens de grandeza, necessárias para
  séries normalizadas. Tay & Cao (2001) usam {10, 100, 1000}; Kim (2003) expande
  para {0.1, ..., 1000}.
- **epsilon = {0.001, 0.01, 0.1, 0.5}** — união das faixas de Tay & Cao (2001) e
  Kim (2003), cobrindo de dados bem-comportados a séries ruidosas. Cherkassky &
  Ma (2004) sugerem epsilon ~ 3·σ_y·√(ln n / n), que cai nesse intervalo para
  séries mensais normalizadas.
- **gamma = "scale"** — como o pool passa pelo StandardScaler (Var(X) ~ 1),
  gamma ~ 1/n_features. Hsu et al. (2016) recomendam escalar antes de buscar
  gamma, então fixar em "scale" dispensa busca adicional.

## MLP (30 combinações)

- **hidden_layer_sizes = {(10,), (25,), (50,), (100,), (50, 25)}** — de uma
  camada rasa a uma de duas camadas (Zhang 2003; Göçken et al. 2016; Makridakis
  et al. 2020 usam (50,) para séries mensais).
- **activation = {"tanh", "relu"}** — tanh é clássica para dados normalizados
  (Zhang 2003); ReLU acelera a convergência (Glorot, Bordes & Bengio 2011).
- **alpha (L2) = {0.0001, 0.001, 0.01}** — weight decay na faixa de Göçken et al.
  (2016) e Makridakis et al. (2020).
- **Adam, learning_rate_init = 0.001, max_iter = 500, early stopping** se o pool
  tiver ≥ 20 amostras — lr padrão de Kingma & Ba (2015); Adam é mais robusto que
  SGD em pools heterogêneos.

## Referências

- Tay, F. E. H.; Cao, L. (2001). Application of support vector machines in
  financial time series forecasting. *Omega*, 29(4), 309–317.
  https://doi.org/10.1016/S0305-0483(01)00026-3
- Kim, K.-J. (2003). Financial time series forecasting using support vector
  machines. *Neurocomputing*, 55(1–2), 307–319.
  https://doi.org/10.1016/S0925-2312(03)00372-2
- Hsu, C.-W.; Chang, C.-C.; Lin, C.-J. (2016). A practical guide to support
  vector classification. *Technical Report*, LIBSVM.
  https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
- Cherkassky, V.; Ma, Y. (2004). Practical selection of SVM parameters and noise
  estimation for SVM regression. *Neural Networks*, 17(1), 113–126.
  https://doi.org/10.1016/S0893-6080(03)00169-2
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural
  network model. *Neurocomputing*, 50, 159–175.
  https://doi.org/10.1016/S0925-2312(02)00373-1
- Göçken, M.; Özçalici, M.; Boru, A.; Dosdogru, A. T. (2016). Integrating
  metaheuristics and artificial neural networks for improved stock price
  prediction. *Expert Systems with Applications*, 44, 320–331.
  https://doi.org/10.1016/j.eswa.2015.09.029
- Makridakis, S.; Spiliotis, E.; Assimakopoulos, V. (2020). Statistical and
  Machine Learning forecasting methods: Concerns and ways forward.
  *International Journal of Forecasting*, 36(1), 54–74.
  https://doi.org/10.1016/j.ijforecast.2019.04.014
- Glorot, X.; Bordes, A.; Bengio, Y. (2011). Deep sparse rectifier neural
  networks. *Proceedings of AISTATS*, 15, 315–323.
  https://proceedings.mlr.press/v15/glorot11a.html
- Kingma, D. P.; Ba, J. (2015). Adam: A method for stochastic optimization.
  *ICLR*. https://arxiv.org/abs/1412.6980
