## Simulador em desenvolvimento.

Até então, há um corpo de simulador em desenvolvimento, o mesmo utiliza-se de células circulares e calcula para cada atualização do sistema a posição dos usuários (TM) e os valores de
Sombreamento e Desvanecimento Rápido.

<h2> Comentário do Código <a href="https://github.com/BPalhano/Simulador_Antenas/blob/main/Simulador/simulator.py"><code>Simulador</code></a> </h2>

<h3> <code> def antenas(num, dist): </code> </h3>

<p>
 <pre>
 <code>
    x = R*math.sqrt(random.rand()) * random.rand() * 2 * math.pi
    y = R*math.sqrt(random.rand()) * random.rand() * 2 * math.pi
 </code>
 </pre>
 
 X e Y definem as coordenadas cartesianas geradas aleatóriamente para o posicionamento do usuários (TM)
 
  <pre>
  <code>
      while (contador != num):

         x = R*math.sqrt(random.rand()) * random.rand() * 2 * math.pi
         y = R*math.sqrt(random.rand()) * random.rand() * 2 * math.pi

         arr1 = np.array([dist* math.cos((2*math.pi/num) * contador) , dist* math.sin((2*math.pi/num) * contador)])
         arr2 = np.array([arr1[0] + x, arr1[1] + y])
        
         vetor1 = np.vstack((vetor1, arr1))
         vetor2 = np.vstack((vetor2,arr2))
  </code>
  </pre>
  
  Laço de formação dos vetores de ERB<sub>i,j</sub> e TM<sub>i,j</sub>.
  
</p>
  
<h3> <code> def lognorm(sigma): </code> </h3>
  
  <p>
 <code>x = random.rand() * sigma</code>
 Função de retorno de uma variável log-normalizada.
 
 </p>
 
<h3> <code> dray(sigma, mi): </code> </h3>

<p>
 
 <pre>
 <code>
    x = random.normal(loc=mi, scale=sigma)
    y = random.normal(loc=mi, scale=sigma)

    h = abs(x**2 - y**2) #  |H(i,j)| ^2
  </code>
  </pre>
  
  Função retorna a variável h, uma variável criada a partir da distribuição de Rayleigh.
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

</p>



<h3> <code> def Pot_linear(vet1): </code> </h3>

<p>
 <pre><code>
     vet1 = vet1.copy()

    for i in range(len(vet1)):

        vet1[i] = dB_to_linear(164.8 + np.log10(vet1[i]))

    return vet1  #  em W
  </pre></code>
  A função de ganho de potência linear recebe a matriz de distâncias d<sub>i,j</sub> e copia a mesma para modificação, gerando o vetor P<sub>i,j</sub><br>
  </p>
  
<h3> <code> def Sombreamento(vet1,sd): </code> </h3>
  
<p>
 <pre><code>
   aux = vet1.copy()

    for i in range(len(vet1)):

        aux[i] = dB_to_linear(lognorm(sd))

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

        aux[i] = dB_to_linear(sd*dray(sd,0))

    return aux # em W
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
  </pre></code>
  
   O vetor auxiliar copia o vetor passado( vetor de ganho de potência linear) e substitui na copia os valores para os valores de ganho de potência gerando
  um vetor G<sub>i,j</sub>
</p>


<h3> <code> def SINR(num, vet1, K): </code> </h3>

<p>
 <pre><code>
 
    matrix = vet1.copy()
    trace = np.array([])

    contador = 0
    for element in range(len(matrix)):

        if (contador % (num+1) == 0):

            trace = np.hstack((trace, matrix[element]))
        contador += 1

    Sum_trace = Soma(trace)
    Sum_matrix = Soma(matrix)

    Sum_const = Sum_matrix - Sum_trace

    SNR = np.array([])

    for element in range(len(trace)):

        aux = (trace[element] / (Sum_const + K))

        SNR = np.hstack((SNR,aux))
 </code></pre>
 
 Função geradora da matriz SINR<sub>i,i</sub> a partir dos valores de G<sub>i,j </sub>
 
 
<hr>

<p>
 A partir da matriz SINR<sub>i,i </sub> será criado um dataframe utilizando a biblioteca <code>pandas</code> e será plotado por meio<br>
 da biblioteca <code>matplotlib</code> a função da Empirical Cumulative Distribuited Funcition (eCDF).
</p>
            
<hr>

<h3>Fontes teóricas:</h3>


<ul></ul>
 <li>SAUNDERS, S. R.; ARAGÓN-ZAVALAA. Antennas and propagation for wireless communication systems. [s.l.] Chichester Wiley, 2007.

 <!-- 

Adicionar a referência bibliográfica do Comunicação Móvel Celular.

-->

## Futuras Atualizações:

 - Modificar a formatação das células para hexagonal.
 - Optimizar computacionalmente a velocidade do laço principal.
 - Implementar uma forma inteligente de posicionar as antenas.


