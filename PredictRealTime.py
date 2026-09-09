import subprocess as sp
import threading
import pickle as pkl
from ugot import ugot
from typing import Any, List

import numpy as np
import pandas as pd

# ip="192.168.120.121"
# robot=ugot.UGOT()
# robot.initialize(ip)

nombremodelo="[insert .pkl file name]"
datosA = ["800"] * 125
datosB = ["800"] * 125
datosC = ["800"] * 125
datosD = ["800"] * 125
n=125
procesar = -1
Hz = np.fft.fftfreq(n, 1 / 125)
with open(nombremodelo, "rb") as f:
     rf = pkl.load(f)

def predict(datosA,datosB,datosC,datosD):
    dfA = list(map(float, datosA))
    dfB = list(map(float, datosB))
    dfC = list(map(float, datosC))
    dfD = list(map(float, datosD))
    dfA = np.abs(np.fft.fft(dfA)) ** 2
    dfB = np.abs(np.fft.fft(dfB)) ** 2
    dfC = np.abs(np.fft.fft(dfC)) ** 2
    dfD = np.abs(np.fft.fft(dfD)) ** 2
    ThetaA=np.sum(dfA[(Hz >= 4) & (Hz < 8)])
    ThetaB=np.sum(dfB[(Hz >= 4) & (Hz < 8)])
    ThetaC=np.sum(dfC[(Hz >= 4) & (Hz < 8)])
    ThetaD=np.sum(dfD[(Hz >= 4) & (Hz < 8)])
    AlphaA=np.sum(dfA[(Hz >= 8) & (Hz < 13)])
    AlphaB=np.sum(dfB[(Hz >= 8) & (Hz < 13)])
    AlphaC=np.sum(dfC[(Hz >= 8) & (Hz < 13)])
    AlphaD=np.sum(dfD[(Hz >= 8) & (Hz < 13)])
    BetaA=np.sum(dfA[(Hz >= 13) & (Hz < 30)])
    BetaB=np.sum(dfB[(Hz >= 13) & (Hz < 30)])
    BetaC=np.sum(dfC[(Hz >= 13) & (Hz < 30)])
    BetaD=np.sum(dfD[(Hz >= 13) & (Hz < 30)])
    GammaA=np.sum(dfA[(Hz >= 30)])
    GammaB=np.sum(dfB[(Hz >= 30)])
    GammaC=np.sum(dfC[(Hz >= 30)])
    GammaD=np.sum(dfD[(Hz >= 30)])
    predictData = dict(zip(['ThetaA', 'ThetaB', 'ThetaC', 'ThetaD',
                          'AlphaA', 'AlphaB', 'AlphaC', 'AlphaD',
                          'BetaA', 'BetaB', 'BetaC', 'BetaD',
                          'GammaA', 'GammaB', 'GammaC', 'GammaD'],
                         [ThetaA, ThetaB, ThetaC, ThetaD,
                          AlphaA, AlphaB, AlphaC, AlphaD,
                          BetaA, BetaB, BetaC, BetaD,
                          GammaA, GammaB, GammaC, GammaD]))
    df=pd.DataFrame(predictData, index=[0])
    precision=rf.predict_proba(df)
    return rf.predict(df)

COLOR_VERDE = '\033[92m'
COLOR_ROJO = '\033[91m'
RESET = '\033[0m'
while True:
    process = sp.Popen(["ConDumpAuto.exe"], stdout=sp.PIPE, text=True, shell=False)
    for line in iter(process.stdout.readline, ""):
        if line.startswith("-") or line[0].isdigit():
            line = line.replace('\r', '').replace('\n', '')
            current = line.split(",")
            procesar = procesar + 1
            datosA[procesar]=current[0]
            datosB[procesar]=current[1]
            datosC[procesar]=current[2]
            datosD[procesar]=current[3]
        if procesar == 64:
            resp1=predict(datosA,datosB,datosC,datosD)[0]
            if resp1 == "Adelante(Der)":
                print(f"{COLOR_VERDE}ADELANTE{RESET}")
                #robot.spider_move_speed(0,5)
            else:
                print(f"{COLOR_ROJO}ATRAS{RESET}")
                #robot.spider_move_speed(1,5)
        elif procesar == 124:
            resp1=predict(datosA,datosB,datosC,datosD)[0]
            if resp1 == "Adelante(Der)":
                print(f"{COLOR_VERDE}ADELANTE{RESET}")
                #robot.spider_move_speed(0,5)
            else:
                print(f"{COLOR_ROJO}ATRAS{RESET}")
                #robot.spider_move_speed(1,5)
  
            procesar = -1