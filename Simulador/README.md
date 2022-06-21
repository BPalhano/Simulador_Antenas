## Simulador em desenvolvimento.

Até então, há um corpo de simulador em desenvolvimento, o mesmo utiliza-se de células circulares e calcula para cada atualização do sistema a posição dos usuários (TM) e os valores de
Sombreamento e Desvanecimento Rápido.

<h2> Comentário do Código <a href="https://github.com/BPalhano/Simulador_Antenas/blob/main/Simulador/simulator.py"><code>Simulador</code></a> </h2>

<h3>Bibliotecas Utilizadas:</h3>
<p>
 <pre>
 Numpy;
 Pandas;
 Scipy(Stats);
 Matplotlib(Pyplot);
 Seaborn;
 </pre>
<p>
<h2>FUNÇÕES </h2>
<h3> <code> def ERB_TM(num, dist): </code> </h3>

<p>
 <pre>
 <code>
    ERB = np.array([0, 0, 100])
    ap = dist * math.sqrt(3)/2
    TM = np.array([ap * random.uniform(0, 1), ap * random.uniform(0, 1), 0])
 </code>
 </pre>
 
 Posição da primeira Estação Rádio-Base(ERB) definida, bem como o apótema da célula hexagonal e a posição do primeiro usuário (relativo a ERB<sub>0</sub>)
 
  <pre>
  <code>
    while contador_ERB != 6:
        arr1 = np.array(
            [2*ap * math.cos((math.pi / 6) + contador_ERB*math.pi/3), 2*ap * math.sin((math.pi / 6) +
                                                                                      contador_ERB*math.pi/3), 100])
        ERB = np.vstack((ERB, arr1))

        arr2 = np.array(
            [arr1[0] + dist * random.uniform(0, 1), arr1[1] + dist * random.uniform(0, 1), arr1[2] +
             100 * random.uniform(-1, 0)])
        TM = np.vstack((TM, arr2))

        contador_ERB += 1
  </code>
  </pre>
  
  Laço de posicionamento das ERB<sub>i,j</sub> e dos primeiros 6 TM<sub>i,j</sub>.
  
  <pre>
  <code>
  
      if N > contador_ERB:
        contador_TM = 0

        while contador_TM != num:
            aux = np.array([6 * dist * random.normal(0, 1), 6 * dist * math.sin(math.pi / 3) * random.uniform(0, 1),
                            100 * random.uniform(0, 1)], dtype=object)

            TM = np.vstack((TM, aux))
            contador_TM += 1

    return ERB, TM
  </code>
  </pre>
  
  Caso haja mais usuários do que ERBs, este trecho é responsável por posicionar os usuários restantes.
  
</p>
  

<h3> <code> def matriz_dist(vet1, vet2): </code></h3>

<p>
 <pre>
 <code>
    for i in vet1:
        
      for j in vet2:

            dist = math.sqrt( pow( (i[0] - j[0]), 2 ) + pow((i[1] - j[1]), 2 ) )
            d = np.hstack((d, dist))
            
  </code>
  </pre>

 Função de construção da matriz de distâncias de cada ERB<sub>i</sub> para todos os TM<sub> i,j </sub>
</p>

<h3> <code> def linear_to_dB(x): </code> e <code> def linear_to_dB(x): </code> </h3>

<p>
 Funções para conversar de escala linear para escala decibel e vice e versa.
 
 <code>x = 10 **(x/10)</code> para o dB_to_linear<br>
 <code>x = 10* np.log10(x)</code> para o linear_to_dB<br>
 <code> x = 10 ** (x/10 -3) </code> ára o dBm_to_linear<br>

</p>



<h3> <code> def Path_loss(vet1): </code> </h3>

<p>
 <pre><code>
    vet1 = vet1.copy()

    for i in range(len(vet1)):
        vet1[i] = dB_to_linear(20 * np.log10(4 * math.pi * vet1 / _lambda))

    return vet1  # em W
  </pre></code>
  A função de ganho de potência linear recebe a matriz de distâncias d<sub>i,j</sub> e copia a mesma para modificação, gerando o vetor P<sub>i,j</sub><br>
  </p>
  
<h3> <code> def Sombreamento(vet1,sd): </code> </h3>
  
<p>
 <pre><code>
    aux = vet1.copy()

    rv = random.lognormal(mean=0, sigma=sd)

    while rv > 100:
        rv = random.lognormal(mean=0, sigma=sd)

    for i in range(len(vet1)):
        aux[i] = dB_to_linear(rv)

    return aux  # em W
 </pre></code>
  O vetor auxiliar copia o vetor passado( vetor de ganho de potência linear) e substitui na copia os valores para os valores de sombreamento gerando
  um vetor S<sub>i,j</sub>
</p>

<h3><code>def Fast_fadding(vet1, sd):</code></h3>

<p>
 <pre><code>
    aux = vet1.copy()

    for i in range(len(vet1)):
        rv = scs.rice.rvs(10)
        aux[i] = dB_to_linear(rv)

    return aux  # em W
  </pre></code>
   O vetor auxiliar copia o vetor passado( vetor de ganho de potência linear) e substitui na copia os valores para os valores de desvanecimento rápido gerando
  um vetor D<sub>i,j</sub> 
  
</p>
 
<h3><code>def Ganho(vet1,vet2,vet3, K): </code></h3>
 
<p>
  <pre><code>
          G = vet1.copy()

    for i in range(len(vet1)):
        G[i] = vet1[i] * vet2[i] * vet3[i] * K

    return G  # em W
  </pre></code>
  
   O vetor auxiliar copia o vetor passado( vetor de ganho de potência linear) e substitui na copia os valores para os valores de ganho de potência gerando
  um vetor G<sub>i,j</sub>
</p>


<h3> <code> def SINR(num, vet1, K): </code> </h3>

<p>
 <pre><code>
 
    matrix = vet1.copy()
    trace = 0

    for i in range(0, 7):
        trace += matrix[7 * i]

    tot = Soma(matrix)

    cnt = tot - trace

    SNR = np.array([])

    for i in range(7):
        val = matrix[i * 7] / (const + cnt)

        SNR = np.hstack((SNR, val))

    return SNR
 </code></pre>
 
 Função geradora da matriz SINR<sub>i,i</sub> a partir dos valores de G<sub>i,j </sub>
 
 
<hr>
<h3> <code> def eCDF(nvet1): </code> </h3>
<p>
<pre><code>
    dataset = vet1.copy()
    df = pd.DataFrame(dataset)

    sbn.ecdfplot(df)
 </code></pre>
 
 Função para gerar o plot da Cumulative Distribuition Function (CDF) dos valores de ruído obtido pela simulação.
</p>
            
<hr>

<h3>Fontes teóricas:</h3>


<ul></ul>
 <li>SAUNDERS, S. R.; ARAGÓN-ZAVALAA. Antennas and propagation for wireless communication systems. [s.l.] Chichester Wiley, 2007.
 <li>YONG SOO CHO et al. MIMO-OFDM wireless communications with MATLAB. Singapore ; Hoboken, Nj: Ieee Press, 2011.
 <li>Özlem Tuğfe Demir, Emil Björnson and Luca Sanguinetti (2020),“Foundations of User-centric Cell-free Massive MIMO”,
‌

 <!-- 

Adicionar a referência bibliográfica do Comunicação Móvel Celular.

-->

## Futuras Atualizações:

 - Implementar uma forma inteligente de posicionar as antenas.
 - Implementar classe de usuários (métodos: Sombreamento, célular associada, Visada, Desvanecimento Rápido, Distância a base).
 - Implementar intereções do sinal na camada física (cenário Cell-Free Massivo MIMO).

