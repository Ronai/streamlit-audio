# -- Usar o backend Agg para garantir segurança em ambientes com múltiplas threads.
import matplotlib as mpl
mpl.use("agg")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.api import fetch_event_json
from copy import deepcopy
import io
from scipy import signal
from scipy.io import wavfile
from freqdomain2 import showfreqdomain

# -- Funções auxiliares deste repositório Git.
from helper import *

apptitle = 'Tutorial de Processamento de Sinais'

st.set_page_config(page_title=apptitle, page_icon=":headphones:",
                               initial_sidebar_state='collapsed')

# Definir o título do aplicativo.
st.title(apptitle)

fs = 32000
noisedt = 8
noise = deepcopy(makewhitenoise(fs, noisedt))

#-- Tentar colorir o ruído
noisefreq = noise.fft()
color = 1.0 / (noisefreq.frequencies)**2
indx = np.where(noisefreq.frequencies.value < 30)
color[indx] = 0  #-- Aplicar o corte de baixa frequência em 30 Hz

#-- Ruído vermelho no domínio da frequência
weightedfreq = noisefreq * color.value

# -- Tentar retornar ao domínio do tempo.
colorednoise = weightedfreq.ifft()

###
# -- Injetar o sinal.
###
secret = TimeSeries.read('LOZ_Secret.wav')

# -- Normalizar e converter para ponto flutuante.
secret -= secret.value[0]  #-- Remover o deslocamento constante.
secret = np.float64(secret)
secret = secret/np.max(np.abs(secret)) * 1*1e-8   #-- Definir a amplitude.
secret.t0 = 4

volume = st.sidebar.radio("Volume do som secreto", ["Padrão", "Mais alto"])

if volume == 'Mais alto':
    maze = colorednoise.inject(10*secret)
else:
# -- Pode ser útil incluir uma opção para facilitar a audição.
    maze = colorednoise.inject(secret)


# -------
# Iniciar a exibição a partir daqui.
# -------
st.markdown("## Introdução")

st.markdown("""
Nesta demonstração, tentaremos encontrar um **som secreto** escondido em dados ruidosos. 
Para isso, vamos explorar alguns conceitos de processamento de sinais:

 * Representação gráfica nos domínios do tempo e da frequência
 * Filtragem passa-altas e passa-faixa
 * Branqueamento (whitening)
""")

sectionnames = [
                'Introdução ao domínio da frequência',
                'Ruído branco (White Noise)',
                'Ruído vermleho (Red Noise)',
                'Encontre o som secreto',
                'Branqueamento (Whitening)',
                'Dados de ondas gravitacionais',
]

def headerlabel(number):
    return "{0}: {1}".format(number, sectionnames[number-1])
    
page = st.radio('Selecionar seção:', [1,2,3,4,5,6], format_func=headerlabel)

st.markdown("## {}".format(headerlabel(page)))

if page==1:
    
    showfreqdomain()
    
if page==2:

    # Ruído branco
    
    st.markdown("""
    Agora, vamos observar um pouco de **ruído branco**. 
    Qualquer sinal pode ser representado a partir de seu conteúdo em frequência. 
    Quando dizemos que um ruído é branco, queremos dizer que o sinal apresenta aproximadamente 
    a mesma amplitude em todas as frequências. 
    
    A seguir, representaremos o **mesmo sinal de três maneiras diferentes**:
    
    *Um sinal no domínio do tempo  
    *Um sinal no domínio da frequência  
    *Um arquivo de áudio
    """)

    st.markdown("### Domínio do tempo")

    st.markdown("""
    No **domínio do tempo**, observamos um sinal como função do tempo. 
    O eixo x representa o tempo, enquanto o eixo y representa o valor do sinal em cada instante. 
    Para um sinal de áudio, esse valor corresponde à variação de pressão percebida pelo tímpano em cada momento. 
    Para um sinal de **onda gravitacional**, o valor do sinal representa o strain, isto é, a variação fracionária 
    do comprimento dos braços do observatório.
    """)

    tplot = noise.plot(ylabel='Pressão')
    st.pyplot(tplot)
    
    st.markdown("### Domínio da frequência")

    st.markdown("""
    No **domínio da frequência**, o eixo x representa os valores de frequência, 
    enquanto o eixo y mostra a **amplitude** — ou a grandeza estreitamente relacionada, 
    a densidade espectral de amplitude — do sinal em cada frequência. 
    Como o ruído branco apresenta aproximadamente a mesma amplitude em todas as frequências, 
    esse gráfico é praticamente plano ao longo do eixo das frequências.
    """)

    figwn = noise.asd(fftlength=1).plot(ylim=[1e-10, 1], ylabel='Densidade espectral de amplitude')
    st.pyplot(figwn)

    st.markdown("### Reprodutor de áudio")
    st.markdown("""
    :point_right: **Use o reprodutor de áudio para ouvir o sinal. 
    Você deverá ouvir o chiado característico do ruído branco.**.
    """)
    
    st.audio(make_audio_file(noise), format='audio/wav')

    st.markdown("")
    st.markdown("""
    Quando estiver pronto, vá para a próxima seção usando os controles na parte superior.
    """)
    
if page == 3:

    # st.markdown("## 3: Ruído vermelho")
    
    st.markdown("""
    Agora, vamos observar um pouco de ruído vermelho. O ruído vermelho apresenta mais potência 
    em baixas frequências do que em altas frequências.
    
    Pode ser difícil imaginar como se comporta um ruído aleatório distribuído em diferentes frequências. 
    Uma forma divertida de visualizar isso é pensar em um estádio esportivo cheio de animais fazendo barulho. 
    Alguns animais, como pássaros e gatinhos, produzem sons mais agudos, enquanto outros, como sapos e leões, 
    produzem sons mais graves. Se o estádio tiver animais de todos os tipos em quantidades semelhantes, o 
    resultado poderia se parecer com um ruído branco. Já se o estádio estiver cheio de animais que produzem 
    ons graves — por exemplo, muitos sapos — o resultado poderia se parecer com um ruído vermelho. 
    Você consegue imaginar a diferença?
    
    Uma ideia semelhante pode ser observada no ruído dos detectores LIGO e Virgo. 
    Fontes de ruído de baixa frequência contribuem principalmente nas baixas frequências. 
    Em geral, estão associadas a estruturas grandes e de movimento lento, especialmente ao movimento contínuo 
    do solo, conhecido como movimento sísmico. Em frequências mais altas, há diversas fontes de ruído 
    associadas à vibração de componentes do instrumento, como espelhos e mesas ópticas.
    """)

    ###
    # -- Mostrar o ruído vermelho com o sinal
    ###

    st.markdown("No domínio do tempo, podemos observar que o ruído vermelho apresenta um comportamento aleatório.")

    figrnt = maze.plot(ylabel='Pressão')
    st.pyplot(figrnt)

    st.markdown("No domínio da frequência, o ruído vermelho apresenta maior potência nas baixas frequências.")

    figrn = maze.asd(fftlength=1).plot(ylabel='Densidade espectral de amplitude', ylim=[1e-11, 1e-4], xlim=[30, fs/2])
    st.pyplot(figrn)
        
    st.audio(make_audio_file(maze), format='audio/wav')
    st.markdown("""
    Você consegue ouvir os sapos coaxando?

    :point_right: **Como esse som se compara ao som do ruído branco?**
    """)

if page == 4:

    # ----
    # Tente recuperar o sinal
    # ----
    # st.markdown("## 4: Encontre o som secreto")
    
    st.markdown("""
    O ruído vermelho acima não é apenas ruído — há um som secreto escondido nele. 
    Você conseguiu ouvi-lo? Provavelmente não! Todo esse ruído de baixa frequência está tornando o 
    som secreto muito difícil de perceber. Mas... se o som secreto estiver em frequências mais altas, 
    talvez ainda seja possível ouvi-lo.

    O que precisamos é de uma forma de remover parte do ruído de baixa frequência, preservando 
    a parte do sinal em frequências mais altas. Em processamento de sinais, isso é feito com um filtro 
    passa-altas (high-pass filter): um filtro que atenua os sons de baixa frequência e mantém, ou 
    permite passar, os sons de alta frequência. A frequência de corte (cutoff frequency) define essa separação: 
    frequências abaixo dela são atenuadas, enquanto frequências acima dela são preservadas.

    Veja se você consegue usar um filtro passa-altas para encontrar o som secreto. 

    :point_right: **Ajuste a frequência de corte usando o controle deslizante abaixo e veja se é 
     possível remover parte do ruído e revelar o som secreto.**

    """)

    lowfreq = st.slider("Frequência de corte do filtro passa-altas (Hz)", 0, 3000, 0, step=100)
    if lowfreq == 0: lowfreq=1

    highpass = maze.highpass(lowfreq)
    #st.pyplot(highpass.plot())

    fighp = highpass.asd(fftlength=1).plot(ylabel='Densidade espectral de amplitude',
                                           ylim=[1e-12, 1e-5],
                                           xlim=[30, fs/2]
                                           )
    ax = fighp.gca()
    ax.axvspan(1, lowfreq, color='red', alpha=0.3, label='Removido pelo filtro')
    st.pyplot(fighp)

    st.audio(make_audio_file(highpass), format='audio/wav')

    st.markdown("Você consegue ouvir o som agora? Qual valor da frequência de corte torna o som mais fácil de ouvir?")

    st.markdown("")
    needhint = st.checkbox("Precisa de uma dica?", value=False)

    if needhint:

        st.markdown("""Aqui está o som secreto. Você consegue identificá-lo 
        escondido no ruído vermelho acima?
        """)

        st.audio(make_audio_file(secret), format='audio/wav')

        st.markdown("""Você também pode facilitar a audição do som selecionando 
        a opção 'Mais alto' no menu à esquerda.
        """)
        
if page == 5:
    # st.markdown("## 5: Branqueamento")

    st.markdown("""
    O **branqueamento** (whitening) é um processo que repondera o sinal de modo que todas as 
    faixas de frequência apresentem aproximadamente a mesma quantidade de ruído. 
    Em nosso exemplo, é difícil ouvir o sinal porque o ruído de baixa frequência acaba encobrindo-o. 
    Ao aplicar o branqueamento aos dados, podemos evitar que o ruído de baixa frequência domine aquilo que ouvimos.
    
    :point_right: **Use a caixa de seleção para aplicar o branqueamento aos dados.**
    """)

    
    whiten = st.checkbox("Aplicar branqueamento aos dados?", value=False)

    if whiten:
        whitemaze = maze.whiten()
    else:
        whitemaze = maze

    st.markdown("""
    Após o branqueamento, você pode observar o som secreto no **domínio do tempo**. 
    Também é possível notar que o sinal branqueado surge gradualmente no início e desaparece suavemente no final. 
    Essa transição gradual de entrada e saída é causada pelo **janelamento** (windowing), 
    uma técnica importante em muitas aplicações de processamento de sinais.
    """)
    
    st.pyplot(whitemaze.plot())

    figwh = whitemaze.asd(fftlength=1).plot(ylim=[1e-12, 1], xlim=[30,fs/2], ylabel='Densidade espectral de amplitude')
    st.pyplot(figwh)
    
    st.audio(make_audio_file(whitemaze), format='audio/wav')

    st.markdown("""Experimente usar a caixa de seleção para aplicar o branqueamento aos dados. 
    É mais fácil ouvir o som secreto com ou sem o branqueamento?
    """)

if page == 6:

    # st.markdown("## 6: Dados de ondas gravitacionais")

    st.markdown("""
    Por fim, vamos aplicar o que aprendemos a **dados reais de ondas gravitacionais do LIGO**, 
    em torno do sinal **GW150914**, produzido pela coalescência de um sistema binário de buracos negros. 
    Vamos acrescentar mais um elemento: um **filtro passa-faixa* ('band-pass filter'). 
    Esse filtro utiliza uma frequência de corte inferior e uma frequência de corte superior, 
    permitindo a passagem apenas das componentes do sinal cujas frequências estejam dentro desse intervalo.

    :point_right: **Experimente usar um **filtro de branqueamento** e um **filtro passa-faixa** para revelar 
    o sinal de onda gravitacional nos dados abaixo.**  
    """)

    detector = 'H1'
    t0 = 1126259462.4   #-- GW150914

    st.text("Detector: {0}".format(detector))
    st.text("Time: {0} (GW150914)".format(t0))
    strain = load_gw(t0, detector)
    center = int(t0)
    strain = strain.crop(center-14, center+14)

    # -- Experimente visualizar os dados após o branqueamento e a filtragem passa-faixa.
    # -- Aplicar branqueamento e filtragem passa-faixa aos dados.
    st.subheader('Dados branqueados e filtrados por passa-faixa')

    lowfreqreal, highfreqreal = st.slider("Frequências de corte do filtro passa-faixa (Hz)",
                                          1, 1200, value=(1,1200) )

    makewhite = st.checkbox("Aplicar branqueamento", value=False)

    if makewhite:
        white_data = strain.whiten()
    else:
        white_data = strain

    bp_data = white_data.bandpass(lowfreqreal, highfreqreal)

    st.markdown("""
    Com a filtragem adequada, talvez seja possível visualizar o sinal no gráfico no domínio do tempo.
    """)

    fig3 = bp_data.plot(xlim=[t0-0.1, t0+0.1])
    st.pyplot(fig3)

    # -- Densidade espectral de potência (PSD) dos dados branqueados
    # -- Plotar a PSD
    psdfig = bp_data.asd(fftlength=4).plot(xlim=[10, 1800], ylabel='Densidade espectral de amplitude')    
    ax = psdfig.gca()
    ax.axvspan(1, lowfreqreal, color='red', alpha=0.3, label='Removed by filter')
    ax.axvspan(highfreqreal, 1800, color='red', alpha=0.3, label='Removido pelo filtro')
    st.pyplot(psdfig)

    # -- Áudio
    st.audio(make_audio_file(bp_data.crop(t0-1, t0+1)), format='audio/wav')

    # -- Fechar todas as figuras abertas
    plt.close('all')

    st.markdown("""Com a filtragem adequada, talvez seja possível ouvir o sinal produzido pela 
    coalescência dos buracos negros. O som é muito breve — algo semelhante a um pequeno **baque**.  
 """)

    st.markdown("")
    hint = st.checkbox('Precisa de uma dica?')

    if hint:

        st.markdown("""
        Dica: experimente usar um filtro passa-faixa de 30 a 400 Hz, 
        com o branqueamento ativado. Isso é semelhante ao procedimento utilizado na Figura 1 do
        [artigo de descoberta do GW150914](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.061102), também mostrado abaixo:
        """)
        
        st.image('https://journals.aps.org/prl/article/10.1103/PhysRevLett.116.061102/figures/1/large')

st.markdown("""## Sobre este aplicativo

Este aplicativo exibe dados do LIGO, Virgo e GEO obtidos a partir do Gravitational Wave Open Science Center at 
[https://gwosc.org](https://gwosc.org).
""")
