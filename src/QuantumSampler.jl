"""
    BuildInitialState(Parameters)

Use one parameter `gamma` to build the initial distributions.
"""
function BuildInitialState(Parameters)
    alpha = beta = Parameters
    x_all = [.005; .01:.01:.99; .995]
    x0 = map(x -> pdf(Beta(alpha, beta), x), x_all)
    Psy0 = x0 ./ sum(x0)
    return Psy0
end

"""
    GenerateDriftRates(Parameters, prob; N = 101)

First, use two parameters `a` and `b`, and the observed probability `prob` to generate the intensity matrix.
Second, use matrix exponentinal to generate the drift rates.

"""
function GenerateDriftRates(Parameters, prob; N = 101)
    α, b = Parameters
    (c₊, c₋) = b >= 0 ? (1+b, 1) : (1, 1-b)
    β₊ = α *    prob + c₊
    β₋ = α *(1-prob) + c₋
    K = zeros(N, N)
    for idx in CartesianIndices(K)
        (2 <= idx[1] <= N-1) && (K[idx[1],   idx[1]  ] = -(β₊ + β₋))
        (1 <= idx[1] <= N-1) && (K[idx[1],   idx[1]+1] =  β₋) # Definition in Huang 2024, Equation 17 is incorrect
        (1 <= idx[2] <= N-1) && (K[idx[2]+1, idx[2]  ] =  β₊) # Definition in Huang 2024, Equation 17 is incorrect
        (idx[1] == 1)        && (K[idx[1],   idx[1]  ] = -β₊)
        (idx[1] == N)        && (K[idx[1],   idx[1]  ] = -β₋)
    end
    T1 = exp(K) # \phi(t)=e^{K\cdot t}\phi(0).
    return T1
end

"""
  GenerateQuantumPredictions(Parameters)

Use 7 parameters to generate the 78 other probabilites.

- `pA`, `pB`, `pC`
- `pBgA`, `pCgA`, `pCgB`
- `o1`: Intereference parameter

"""
function CalculateQuantumPredictions(Parameters)
    ## true probabilities 
    ####################################
    # A = A1,  B = A2,  C = A3
    pA   = Parameters[1]; pnA = 1 - pA
    pB   = Parameters[2]; pnB = 1 - pB
    pC   = Parameters[3]; pnC = 1 - pC

    pBgA = Parameters[4]; pnBgA = 1 - pBgA
    pCgA = Parameters[5]; pnCgA = 1 - pCgA
    pCgB = Parameters[6]; pnCgB = 1 - pCgB


    ## true probabilities 
    ####################################
    o1 = Parameters[7]
    o3 = o4 = o6 = o7 = o9 = o1
    # Classical models have these bounds
    o2 = pA >= pB ? -o1 : o1
    o5 = pA >= pC ? -o1 : o1
    o8 = pB >= pC ? -o1 : o1

    ## A & B 
    ####################################
    pAB  = pA * pBgA
    pAnB = pA * pnBgA

    # order effects
    pBA = pAB - o1
    # Quantum constrains
    pBA = (pBA <= 0) ? 0 : (pBA < pB) ? pBA : pB
    Ord1 = pAB - pBA

    pnBA = pAnB - o2
    pnBA = (pnBA <= 0) ? 0 : (pnBA < pnB) ? pnBA : pnB
    Ord2 = pAnB - pnBA

    pnBnA = pnB - pnBA
    pBnA = pB - pBA

    # reversed order effects
    pnAnB = pnBnA - o3
    pnAnB = (pnAnB <= 0) ? 0 : (pnAnB < pnA) ? pnAnB : pnA
    Ord3 = pnAnB - pnBnA
    pnAB = pnA - pnAnB

    # conditionals
    pAgB  = pBA / pB;    pnAgB = 1 - pAgB
    pAgnB = pnBA / pnB; pnAgnB = 1 - pAgnB

    # reversed conditionals
    pnBgnA = pnAnB / pnA; pBgnA = 1 - pnBgnA

    # MoreLikelyFirst logic for A and B
    pAandB, pnAandnB = pA > pB  ? (pAB, pnBnA) : (pBA, pnAnB)
    pAandnB, pnAandB = pA > pnB ? (pAnB, pBnA) : (pnBA, pnAB)

    pAorB = 1 - pnAandnB
    pnAorB = 1 - pAandnB
    pAornB = 1 - pnAorB
    pnAornB = 1 - pAandB

    ## A & C 
    ####################################
    pAC = pA * pCgA
    pAnC = pA * pnCgA

    # order effects
    pCA = pAC - o4
    pCA = (pCA <= 0) ? 0 : (pCA < pC) ? pCA : pC
    Ord4 = pAC - pCA
    pnCA = pAnC - o5
    pnCA  = (pnCA <= 0) ? 0 : (pnCA < pnC) ? pnCA : pnC
    Ord5  = pAnC - pnCA
    pnCnA = pnC - pnCA
    pCnA  = pC - pCA

    # reversed order effects
    pnAnC = pnCnA - o6
    pnAnC = (pnAnC <= 0) ? 0 : (pnAnC < pnA) ? pnAnC : pnA
    Ord6 = pnAnC - pnCnA
    pnAC = pnA - pnAnC

    # conditionals
    pAgC = pCA / pC; pnAgC = 1 - pAgC
    pAgnC = pnCA / pnC; pnAgnC = 1 - pAgnC

    # reversed conditionals
    pnCgnA = pnAnC / pnA; pCgnA = 1 - pnCgnA

    # MoreLikelyFirst for A and C
    pAandC, pnAandnC = pA > pC  ? (pAC, pnCnA) : (pCA, pnAnC)
    pAandnC, pnAandC = pA > pnC ? (pAnC, pCnA) : (pnCA, pnAC)

    pAorC = 1 - pnAandnC
    pnAorC = 1 - pAandnC
    pAornC = 1 - pnAorC
    pnAornC = 1 - pAandC

    ## B & C
    ####################################
    pBC = pB * pCgB
    pBnC = pB * pnCgB

    # order effects
    pCB = pBC - o7
    pCB = (pCB <= 0) ? 0 : (pCB < pC) ? pCB : pC

    Ord7 = pBC - pCB
    pnCB = pBnC - o8
    pnCB = (pnCB <= 0) ? 0 : (pnCB < pnC) ? pnCB : pnC

    Ord8 = pBnC - pnCB
    pnCnB = pnC - pnCB
    pCnB = pC - pCB

    # reversed order effects
    pnBnC = pnCnB - o9
    pnBnC = (pnBnC <= 0) ? 0 : (pnBnC < pnB) ? pnBnC : pnB

    Ord9 = pnBnC - pnCnB
    pnBC = pnB - pnBnC

    # conditionals
    pBgC = pCB / pC; pnBgC = 1 - pBgC
    pBgnC = pnCB / pnC; pnBgnC = 1 - pBgnC

    # reversed conditionals
    pnCgnB = pnBnC / pnB; pCgnB = 1 - pnCgnB

    # MoreLikelyFirst for B and C
    pBandC, pnBandnC = pB > pC  ? (pBC, pnCnB) : (pCB, pnBnC)
    pBandnC, pnBandC = pB > pnC ? (pBnC, pCnB) : (pnCB, pnBC)

    pBorC = 1 - pnBandnC
    pnBorC = 1 - pBandnC
    pBornC = 1 - pnBorC
    pnBornC = 1 - pBandC

    ### Pred array
    ####################################
    # A = A1, B = A2, C = A3

    #       1       2      3       4         5       6
    #      P(A1)   P(A2)  P(A3)  P(¬A1)    P(¬A2) P(¬A3)
    Pred = [
        pA,   pB,   pC,   pnA,   pnB,   pnC, 

    #    7      8            9       10
    # P(A2|A1) P(A2|¬A1) P(¬A2|A1) P(¬A2|¬A1)
        pBgA,   pBgnA,   pnBgA,   pnBgnA, 

    #   11         12           13          14
    # P(A1 ∩ A2) P(A1 ∩ ¬A2) P(¬A1 ∩ A2) P(¬A1 ∩ ¬A2)
        pAandB,   pAandnB,   pnAandB,   pnAandnB, 

    #   15          16           17          18
    # P(A1 ∪ A2) P(A1 ∪ ¬A2)  P(¬A1 ∪ A2) P(¬A1 ∪ ¬A2)
        pAorB,   pAornB,   pnAorB,   pnAornB, 

    #    19        20         21        22
    #  P(A3|A2) P(A3|¬A2)  P(¬A3|A2) P(¬A3|¬A2)
        pCgB,   pCgnB,   pnCgB,   pnCgnB, 

    #  23          24              25            26
    # P(A2 ∩ A3) P(A2 ∩ ¬A3)    P(¬A2 ∩ A3)    P(¬A2 ∩ ¬A3)
        pBandC,   pBandnC,   pnBandC,   pnBandnC, 

    #  27               28         29          30
    # P(A2 ∪ A3) P(A2 ∪ ¬A3)  P(¬A2 ∪ A3)   P(¬A2 ∪ ¬A3)
        pBorC,   pBornC,   pnBorC,   pnBornC, 

    #    31         32         33             34
    #  P(A1|A3) P(A1|¬A3)   P(¬A1|A3)      P(¬A1|¬A3)
        pAgC,   pAgnC,   pnAgC,   pnAgnC, 

    #    35          36            37         38
    # P(A3 ∩ A1) P(A3 ∩ ¬A1)    P(¬A3 ∩ A1) P(¬A3 ∩ ¬A1)
        pAandC,   pnAandC,   pAandnC,   pnAandnC, 

    #    39         40          41          42
    # P(A3 ∪ A1) P(A3 ∪ ¬A1) P(¬A3 ∪ A1) P(¬A3 ∪ ¬A1)
        pAorC,   pnAorC,   pAornC,   pnAornC, 

    #  43          44          45          46
    # P(A1|A2)   P(¬A1|A2)    P(A1|¬A2)  P(¬A1|¬A2)
        pAgB,   pnAgB,   pAgnB,   pnAgnB, 

    #   47          48          49             50
    # P(A2 ∩ A1) P(¬A2 ∩ A1) P(A2 ∩ ¬A1)    P(¬A2 ∩ ¬A1)
        pAandB,   pAandnB,   pnAandB,   pnAandnB, 

    #   51           52         53           54
    # P(A2 ∪ A1) P(¬A2 ∪ A1) P(A2 ∪ ¬A1)  P(¬A2 ∪ ¬A1)
        pAorB,   pAornB,   pnAorB,   pnAornB, 

    #    55          56          57         58
    # P(A2|A3)   P(¬A2|A3)    P(A2|¬A3)   P(¬A2|¬A3)
        pBgC,   pnBgC,   pBgnC,   pnBgnC, 

    #   59          60           61            62
    # P(A3 ∩ A2) P(¬A3 ∩ A2)   P(A3 ∩ ¬A2)   P(¬A3 ∩ ¬A2)
        pBandC,   pBandnC,   pnBandC,   pnBandnC, 

    #   63           64               65        66
    # P(A3 ∪ A2) P(¬A3 ∪ A2)    P(A3 ∪ ¬A2) P(¬A3 ∪ ¬A2)
        pBorC,   pBornC,   pnBorC,   pnBornC, 

    #   67       68            69          70
    # P(A3|A1) P(¬A3|A1)     P(A3|¬A1)  P(¬A3|¬A1)
        pCgA,   pnCgA,   pCgnA,   pnCgnA, 

    #    71         72           73         74
    # P(A1 ∩ A3) P(¬A1 ∩ A3) P(A1 ∩ ¬A3) P(¬A1 ∩ ¬A3)
        pAandC,   pnAandC,   pAandnC,   pnAandnC, 

    #    75          76            77            78
    # P(A1 ∪ A3) P(¬A1 ∪ A3)   P(A1 ∪ ¬A3)   P(¬A1 ∪ ¬A3)
        pAorC,   pnAorC,   pAornC,   pnAornC
    ]
    Ords = [Ord1,Ord2,Ord3,Ord4,Ord5,Ord6,Ord7,Ord8,Ord9]
    
    return Pred, Ords
end

"""
QuantumSamplerLikelihood(parm, Sdat)

Use 10 parameters: `P(A)`, `P(B)`, `P(C)`, `P(B|A)`, `P(C|A)`, `P(C|B)`, `o`, `γ`, `a`, `b`,
 and the observed data `Sdat` to compute the loglikelihood value.
"""
function QuantumSamplerLikelihood(parm, Sdat)
    Psy0 = BuildInitialState(parm[9])
    Pred = CalculateQuantumPredictions(parm[[1:6; 10]])[1]
    LL = 0
    for (idx, val) in enumerate(Sdat)
        T1 =GenerateDriftRates(parm[7:8], Pred[idx])
        Psyf = T1 * Psy0
        PR = Psyf[Int(val)+1] # shifted up one because lowest index =1
        LL += log(PR)
    end
    return -2 * LL
end