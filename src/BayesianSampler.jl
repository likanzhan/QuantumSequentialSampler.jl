using Distributions

"""
  BayesianSamplerLikelihood(Parameters, ObservedPercentage; ListCFDF = [11:18; 23:30; 35:42; 47:54; 59:66; 71:78])

- `N`: Sample size in the internal process.
- `P(E)`: Subjective probability of E.
- `S(E) ~ Bin(N, P(E))`
- `F(E) = N - S(E)`
- `P(x|S(E)) ~ Beta(beta + S(E), beta + F(E))`.

"""
function BayesianSamplerLikelihood(Parameters, ObservedPercentage; ListCFDF = [11:18; 23:30; 35:42; 47:54; 59:66; 71:78])
  PredictedProb = CalculateBayesianProbabilities(Parameters[1:6])
  ParameterBeta = Parameters[7]
  N2 = 1  + round(Int, Parameters[9], RoundNearestTiesUp) # conjunctions/disjunctions
  N1 = N2 + round(Int, Parameters[8], RoundNearestTiesUp) # marginals/conditionals
  length(ObservedPercentage) == length(PredictedProb) || error("Length mismatch")
  LL = 0
  for (ObservedIndex, ObservedValue) in enumerate(ObservedPercentage)
    ObservedProb = (ObservedValue == 0) ? 0.005 : (ObservedValue == 100) ? 0.995 : (ObservedValue / 100) # Proportion to percentage
    SampleSize = ObservedIndex ∈ ListCFDF ? N2 : N1
    LL += LogLikelihood(SampleSize, ParameterBeta, ObservedProb, PredictedProb[ObservedIndex])
  end
  return LL
end

"""
  LogLikelihood(SampleSize, ParameterBeta, ObservedProb, PredictedProb)
"""
function LogLikelihood(SampleSize, ParameterBeta, ObservedProb, PredictedProb)
  Likelihood = 0
  for SE in 0:SampleSize
    β₁ = SE + ParameterBeta
    β₂ = (SampleSize - SE) + ParameterBeta
    BetaDist    = Beta(β₁, β₂)
    BinomalDist = Binomial(SampleSize, PredictedProb)
    Avaliable_Values = [0.005; (0.01:0.01:0.99); 0.995]
    BetaNorm    = mapreduce(x -> pdf(BetaDist, x), +, Avaliable_Values)
    PSE  = pdf(BinomalDist, SE)
    PxSE = pdf(BetaDist, ObservedProb) / BetaNorm
    Likelihood += PSE * PxSE
  end
  return -2 * log(Likelihood)
end

"""
  CalculateBayesianProbabilities(parm)

Use six probabilities to calculate 78 other probabilites.
"""
function CalculateBayesianProbabilities(parm)

  # true probabilities
  # A = A1,  B = A2,  C = A3
  pA, pB, pC, pBgA, pCgA, pCgB = parm
  pnA, pnB, pnC, pnBgA, pnCgA, pnCgB = 1 .- (pA, pB, pC, pBgA, pCgA, pCgB)

  # A & B
  pAB = pA * pBgA
  pAnB = pA * pnBgA
  pBA = pAB
  (pBA < 0) || (pBA > pB) && (nLL = 10000)
  pnBA = pAnB
  (pnBA < 0) || (pnBA > pnB) && (nLL = 10000)
  pnBnA = pnB - pnBA
  pBnA = pB - pBA
  pnAnB = pnBnA
  (pnAnB < 0) || (pnAnB > pA) && (nLL = 10000)
  pnAB = pA - pnAnB
  pAgB = pBA / pB
  pnAgB = 1 - pAgB
  pAgnB = pnBA / pnB
  pnAgnB = 1 - pAgnB
  pnBgnA = pnBnA / pnA
  pBgnA = 1 - pnBgnA

  # A & C
  pAC = pA * pCgA
  pAnC = pA * pnCgA
  pCA = pAC
  (pCA < 0) || (pCA > pC) && (nLL = 10000)
  pnCA = pAnC
  (pnCA < 0) || (pnCA > pnC) && (nLL = 10000)
  pnCnA = pnC - pnCA
  pCnA = pC - pCA
  pnAnC = pnCnA
  (pnAnC < 0) || (pnAnC > pA) && (nLL = 10000)
  pnAC = pA - pnAnC
  pAgC = pCA / pC
  pnAgC = 1 - pAgC
  pAgnC = pnCA / pnC
  pnAgnC = 1 - pAgnC
  pnCgnA = pnCnA / pnA
  pCgnA = 1 - pnCgnA

  # B & C
  pBC = pB * pCgB
  pBnC = pB * pnCgB
  pCB = pBC
  (pCB < 0) || (pCB > pC) && (nLL = 10000)
  pnCB = pBnC
  (pnCB < 0) || (pnCB > pnC) && (nLL = 10000)
  pnCnB = pnC - pnCB
  pCnB = pC - pCB
  pnBnC = pnCnB
  (pnBnC < 0) || (pnBnC > pB) && (nLL = 10000)
  pnBC = pB - pnBnC
  pBgC = pCB / pC
  pnBgC = 1 - pBgC
  pBgnC = pnCB / pnC
  pnBgnC = 1 - pBgnC
  pnCgnB = pnCnB / pnB
  pCgnB = 1 - pnCgnB

  Pred = [
    # Define probabilities
    # A = A1,  B = A2,  C = A3
    # 1       2      3       4         5       6
    # P(A1)   P(A2)  P(A3)  P(¬A1)    P(¬A2) P(¬A3)
    pA, pB, pC, pnA, pnB, pnC,

    # 7         8        9         10
    # P(A2|A1) P(A2|¬A1) P(¬A2|A1) P(¬A2|¬A1)      
    pBgA, pBgnA, pnBgA, pnBgnA,

    # 11         12          13          14
    # P(A1 ∩ A2) P(A1 ∩ ¬A2) P(¬A1 ∩ A2) P(¬A1 ∩ ¬A2)
    pAB, pAnB, pnAB, pnAnB,

    # 15          16           17          18
    # P(A1 ∪ A2)  P(A1 ∪ ¬A2)  P(¬A1 ∪ A2) P(¬A1 ∪ ¬A2)
    1 - pnAnB, 1 - pnAB, 1 - pAnB, 1 - pAB,

    # 19        20         21         22
    # P(A3|A2)  P(A3|¬A2)  P(¬A3|A2)  P(¬A3|¬A2)
    pCgB, pCgnB, pnCgB, pnCgnB,

    # 23          24            25            26
    # P(A2 ∩ A3) P(A2 ∩ ¬A3)    P(¬A2 ∩ A3)   P(¬A2 ∩ ¬A3)
    pBC, pBnC, pnBC, pnBnC,

    # 27         28            29           30
    # P(A2 ∪ A3) P(A2 ∪ ¬A3)  P(¬A2 ∪ A3)   P(¬A2 ∪ ¬A3)
    1 - pnBnC, 1 - pnBC, 1 - pBnC, 1 - pBC,

    # 31         32         33             34
    # P(A1|A3) P(A1|¬A3)    P(¬A1|A3)      P(¬A1|¬A3)
    pAgC, pAgnC, pnAgC, pnAgnC,

    # 35          36            37           38
    # P(A3 ∩ A1)  P(A3 ∩ ¬A1)   P(¬A3 ∩ A1)  P(¬A3 ∩ ¬A1)
    pCA, pCnA, pnCA, pnCnA,

    # 39         40           41          42
    # P(A3 ∪ A1) P(A3 ∪ ¬A1)  P(¬A3 ∪ A1) P(¬A3 ∪ ¬A1)
    1 - pnCnA, 1 - pnCA, 1 - pCnA, 1 - pCA,

    # 43          44          45          46
    # P(A1|A2)    P(¬A1|A2)   P(A1|¬A2)   P(¬A1|¬A2)
    pAgB, pnAgB, pAgnB, pnAgnB,

    # 47          48          49             50
    # P(A2 ∩ A1)  P(¬A2 ∩ A1) P(A2 ∩ ¬A1)    P(¬A2 ∩ ¬A1)
    pBA, pnBA, pBnA, pnBnA,

    # 51           52          53           54
    # P(A2 ∪ A1)   P(¬A2 ∪ A1) P(A2 ∪ ¬A1)  P(¬A2 ∪ ¬A1)
    1 - pnBnA, 1 - pBnA, 1 - pnBA, 1 - pBA,

    # 55          56          57          58
    # P(A2|A3)    P(¬A2|A3)   P(A2|¬A3)   P(¬A2|¬A3)
    pBgC, pnBgC, pBgnC, pnBgnC,

    # 59          60            61            62
    # P(A3 ∩ A2)  P(¬A3 ∩ A2)   P(A3 ∩ ¬A2)   P(¬A3 ∩ ¬A2)
    pCB, pnCB, pCnB, pnCnB,

    # 63           64               65         66
    # P(A3 ∪ A2)   P(¬A3 ∪ A2)     P(A3 ∪ ¬A2) P(¬A3 ∪ ¬A2)
    1 - pnCnB, 1 - pCnB, 1 - pnCB, 1 - pCB,

    # 67       68            69         70
    # P(A3|A1) P(¬A3|A1)     P(A3|¬A1)  P(¬A3|¬A1)
    pCgA, pnCgA, pCgnA, pnCgnA,

    # 71         72           73          74
    # P(A1 ∩ A3) P(¬A1 ∩ A3)  P(A1 ∩ ¬A3) P(¬A1 ∩ ¬A3)
    pAC, pnAC, pAnC, pnAnC,

    # 75          76            77            78
    # P(A1 ∪ A3)  P(¬A1 ∪ A3)   P(A1 ∪ ¬A3)   P(¬A1 ∪ ¬A3)
    1 - pnAnC, 1 - pAnC, 1 - pnAC, 1 - pAC
  ]
end