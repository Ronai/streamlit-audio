import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwpy.plot import Plot
from scipy import signal

from helper import makesine, make_audio_file, plot_signal

cropstart = 1.0
cropend   = 1.05


def showfreqdomain():

    st.markdown("""

INTRODUÇÃO

Uma etapa importante em muitos algoritmos de processamento de sinais é transformar dados de série temporal — 
pontos de dados organizados sequencialmente no tempo — em uma nova representação no domínio da frequência.
Neste tutorial, começaremos explicando o que isso significa e por que essa transformação é útil, 
reconstruindo um sinal-alvo a partir de suas componentes.

TRÊS NOTAS

O sinal-alvo abaixo é composto por três alturas sonoras diferentes, ou…
**[frequências](https://youtu.be/Axx8WfxQDkk)**. Imagine que gravamos esse sinal de nossa música 
favorita e queremos descobrir quais são as três frequências utilizadas para produzi-lo. 
Como poderíamos fazer isso? Um problema semelhante aparece em muitos experimentos: 
registramos determinados dados e, em seguida, queremos identificar quais frequências 
contribuíram para gerar o sinal.
""")

    st.markdown("#### Sinal-alvo no domínio do tempo:")

    sig1 = makesine(200, 4, False)
    sig2 = makesine(250, 3, False)
    sig3 = makesine(300, 2, False)
    
    totalsignal = sig1+sig2+sig3
    plot_signal(totalsignal, color_num=1)

    st.audio(make_audio_file(totalsignal), format='audio/wav')

    st.markdown("""
    O gráfico acima mostra o sinal-alvo no **domínio do tempo**. 
    Em um gráfico no domínio do tempo, o eixo x representa sempre o tempo. 
    O eixo y representa a grandeza medida em cada instante de amostragem. 
    Para um sinal sonoro, essa grandeza corresponde à pressão do ar que atinge o ouvido 
    ou o microfone em cada momento.

    Você consegue identificar quais foram as **três frequências**, ou alturas sonoras, 
    usadas para criar esse sinal? Provavelmente não!
    
    Embora o domínio do tempo seja a forma como frequentemente registramos os dados, ele 
    não é a melhor representação para visualizar as frequências que compõem o sinal. 
    Em vez disso, podemos utilizar um processo conhecido como uma
    [Transformada de Fourier](https://www.youtube.com/watch?v=1JnayXHhjlg) para converter o 
    sinal para o **domínio da frequência**.

    :point_right: **Clique na caixa de seleção abaixo para converter o sinal-alvo para o domínio da frequência.**.

    """)

    showfreq = st.checkbox('Converter o sinal-alvo para o domínio da frequência.', value=False)

    if showfreq:
        freqdomain = totalsignal.fft()

        source = pd.DataFrame({
            'Frequência (Hz)': freqdomain.frequencies,
            'Amplitude': np.abs(freqdomain.value),
            'color':['#1f77b4', '#ff7f0e'][1]
        })

        chart = alt.Chart(source).mark_line().encode(
            alt.X('Frequência (Hz)',
                  scale=alt.Scale(
                      domain=(0, 400),
                      clamp=True)),
            alt.Y('Amplitude:Q',
                  scale=alt.Scale(
                      domain=(-0, 5),
                      clamp=True)),
            color=alt.Color('color', scale=None)
        ).properties(title='Sinal-alvo no domínio da frequência')

        st.altair_chart(chart, width='stretch')
            
        st.markdown("""
        A conversão para o **domínio da frequência** permite visualizar as componentes individuais 
        que contribuem para o sinal total. No **domínio da frequência**, a frequência — ou altura sonora — 
        de cada componente do sinal é representada no eixo x.
        A **[amplitude](https://www.youtube.com/watch?v=TsQL-sXZOLc)** 
        (ou intensidade sonora) — de cada componente do sinal é representada no eixo y.

        Usando o gráfico no domínio da frequência acima:
        *Quais são as três frequências utilizadas para compor o sinal total?
        *Qual é a amplitude de cada uma dessas frequências?
        """)


    st.markdown("""
    :point_right: **Tente reconstruir o sinal acima usando três componentes, ou notas. 
                    Você pode ajustar os controles deslizantes para definir cada componente.**.
    """)

    st.markdown("#### Componente 1")
    freq1 = st.slider("Frequência (Hz)", 100, 400, 100, step=10)
    amp1 = st.number_input("Amplitude", 0, 5, 0, key='amp1slider')

    guess1 = makesine(freq1, amp1)
    
    st.markdown("#### Componente 2")
    freq2 = st.slider("Frequência (Hz)", 100, 400, 150, step=10)
    amp2 = st.number_input("Amplitude", 0, 5, 0, key='amp2slider')

    guess2 = makesine(freq2, amp2)
    
    st.markdown("#### Componente 3")
    freq3 = st.slider("Frequência (Hz)", 100, 400, 200, step=10)
    amp3 = st.number_input("Amplitude", 0, 5, 0, key='amp3slider')

    guess3 = makesine(freq3, amp3)

    st.markdown("### Somando as três componentes:")
    
    guess  = guess1 + guess2 + guess3

    chart1 = plot_signal(guess, color_num=0, display=False)
    chart2 = plot_signal(totalsignal, color_num=1, display=False)
    chart = (chart2 + chart1).properties(title='Sinal-alvo (laranja) & tentativa de reconstrução (azul)')
    st.altair_chart(chart, width='stretch')
        
    mismatch = (totalsignal.crop(cropstart, cropend) - guess.crop(cropstart, cropend)).value.max()
    # st.write(mismatch)

    if mismatch < 0.1:
        st.markdown("### Uma correspondência perfeita! Excelente trabalho!!  :trophy:")
        st.balloons()
    elif mismatch < 3:
        st.markdown("### Está bem perto!")    
    
    st.markdown("#### Áudio do sinal-alvo")
    st.audio(make_audio_file(totalsignal), format='audio/wav')

    st.markdown("#### Áudio da tentativa de reconstrução")
    st.audio(make_audio_file(guess), format='audio/wav')
    
    st.markdown("""
    Veja se você consegue recriar o sinal-alvo, ajustando as três componentes.

    *Dica: observe as frequências e amplitudes das componentes no gráfico no domínio da frequência*
    """)

    st.markdown("""
    Quando estiver pronto, vá para a próxima seção usando os controles na parte superior.
    """)
    
    # -- Fechar todas as figuras abertas.
    plt.close('all')
