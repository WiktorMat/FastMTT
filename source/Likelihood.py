import numpy as np
from scipy.constants import physical_constants

def InvariantMass(p4):
    metric = np.array([-1,-1,-1,1])
    p4_square = p4*(metric*p4)
    m = np.sqrt(np.sum(p4_square, axis=-1))
    return m

class Likelihood:
    def __init__(self):
        #METinputs
        self.recoMET = np.array([0.0, 0.0, 0.0, 0.0])
        self.covMET = np.ones((2, 2))

        #setParameters
        self.coeff1 = 6
        self.coeff2 = 1/1.15

        #LeptonInputs
        self.leg1P4 = np.array([0.0, 0.0, 0.0, 0.0])
        self.leg2P4 = np.array([0.0, 0.0, 0.0, 0.0])

        #Visible mass of both leptons
        self.mvis = np.array([0.0, 0.0, 0.0, 0.0])

        #Invariant mass of each lepton
        self.mvisleg1 = np.array([0.0])
        self.mvisleg2 = np.array([0.0])

        self.mVisOverTauSquare1 = np.array([0.0])
        self.mVisOverTauSquare2 = np.array([0.0])
         
        self.mTau = physical_constants['tau energy equivalent'][0]/1000 #MeV -> GeV
        
        self.leg1DecayType = np.array([0.0])
        self.leg2DecayType = np.array([0.0])
        self.leg1DecayMode = np.array([0.0])
        self.leg2DecayMode = np.array([0.0])

        #Enable/disable likelihood channel
        self.enable_MET = True
        self.enable_mass = True

        #Mass constraints of two type -- strict cut (window) or additional likelihood component with normal distribution (mass_constraint)
        self.enable_mass_constraint = False
        self.constraint_mean = 125  # Mass of the particle, that we want to reconstruct. In this case, it is Higgs mass
        
        # For Z0:
        #self.constraint_mean = 91.1876
        self.constraint_sigma = 10 #artificially set value, one can play with it and adjust for the best mass/pt resolution
        #However something around 10GeV seems to work optimally

        self.enable_window = False
        self.window = [123, 127]

        #These are experimental and not used by main code
        self.enable_px = False
        self.enable_py = False

        return

    def setMassConstraint(self, mean, sigma):
        self.constraint_mean = mean
        self.constraint_sigma = sigma

    def setWindow(self, window):
        self.window = window

    def enableLikelihoodComponents(self, MET = None, mass = None, mass_constraint = None, window = None, phi2 = None):  #All Boolean
        if MET is not None:
            self.enable_MET = MET
        if mass is not None:
            self.enable_mass = mass
        if mass_constraint is not None:
            self.enable_mass_constraint = mass_constraint
        if window is not None:
            self.enable_window = window

    def setLeptonInputs(self, aLeg1P4, aLeg2P4, aLeg1DecayType, aLeg2DecayType, aLeg1DecayMode, aLeg2DecayMode):
        
        self.leg1DecayType = aLeg1DecayType
        self.leg2DecayType = aLeg2DecayType
        self.leg1DecayMode = aLeg1DecayMode
        self.leg2DecayMode = aLeg2DecayMode

        self.leg1P4 = aLeg1P4
        self.leg2P4 = aLeg2P4
        
        #visible invariant mass
        #eq. (4)
        self.mvis = InvariantMass(self.leg1P4 + self.leg2P4)
        
        self.mvisleg1[(aLeg1DecayType==1) & (self.mvisleg1>1.5)] = 0.3
        self.mvisleg2[(aLeg2DecayType==1) & (self.mvisleg2>1.5)] = 0.3

        self.mVisOverTauSquare1 = (self.mvisleg1/self.mTau)**2
        self.mVisOverTauSquare2 = (self.mvisleg2/self.mTau)**2

    def massLikelihood(self, m):
        mScaled = m*self.coeff2

        mask1 = (mScaled < self.mvis[:, np.newaxis])
        
        mVS2 = (self.mvis[:, np.newaxis]/mScaled)**2
        
        x1Min = np.minimum(1.0, self.mVisOverTauSquare1)
        x2Min = np.maximum(self.mVisOverTauSquare2[:, np.newaxis], mVS2)
        x2Max = np.minimum(1.0, mVS2/x1Min[:, np.newaxis])
        
        mask2 = (x2Min > x2Max)
        
        jacobiFactor = 2.0*self.mvis[:, np.newaxis]**2*mScaled**(-self.coeff1)
        x2IntegralTerm = np.log(x2Max/x2Min)

        value = 0.0
        value += x2IntegralTerm

        HadDecay1 = np.broadcast_to((self.leg1DecayType != 1)[:, np.newaxis], value.shape)
        value += HadDecay1 * mVS2 * (1 / x2Max - 1 / x2Min)

        HadDecay2 = np.broadcast_to((self.leg2DecayType != 1)[:, np.newaxis], value.shape)
        value += HadDecay2 * (mVS2*x2IntegralTerm - (x2Max - x2Min))

        value[mask1 | mask2] = 0.0

        value *= 1E9*jacobiFactor

        return value

    def mass_constraint(self, invariant_mass):

        Gauss_factor = np.exp(-(invariant_mass - self.constraint_mean)**2/(2*self.constraint_sigma**2))

        return Gauss_factor

    def Window(self, invariant_mass):
        mask = (invariant_mass > self.window[0]) & (invariant_mass < self.window[1])
        return mask
    
    def metTF(self, metP4, nuP4, covMET):
        aMETx = metP4[..., 0]
        aMETy = metP4[..., 1]

        covDET = np.linalg.det(covMET)


        mask = covDET < 1E-10
        covDET[mask] = 1.0

        constMET = 1/2/np.pi/np.sqrt(covDET)
        residualX = aMETx[:, np.newaxis] - nuP4[:, :, 0]
        residualY = aMETy[:, np.newaxis] - nuP4[:, :, 1]

        #covMET 1 coordinate responds to X and 0 coordinate to Y
        pull2 = residualX*(covMET[:, np.newaxis, 1, 1]*residualX - covMET[:, np.newaxis, 0, 1]*residualY) + residualY*(-covMET[:, np.newaxis, 1, 0]*residualX + covMET[:, np.newaxis, 0, 0]*residualY)
        pull2 /= covDET[:, np.newaxis]
        
        pull2[np.broadcast_to(mask[:, np.newaxis], pull2.shape)] = 0.0
        return constMET[:, np.newaxis]*np.exp(-0.5*pull2)
    
    def value(self, x):
        
        x1Min = np.minimum(1.0, self.mVisOverTauSquare1)
        x2Min = np.minimum(1.0, self.mVisOverTauSquare2)

        mask = (x[:, 0] < x1Min[:, np.newaxis]) | (x[:, 1] < x2Min[:, np.newaxis])
        
        testP4 = self.leg1P4[:, np.newaxis, :] / x[:, 0][:, np.newaxis] + self.leg2P4[:, np.newaxis, :] / x[:, 1][:, np.newaxis]

        testMET = testP4 - self.leg1P4[:, np.newaxis, :] - self.leg2P4[:, np.newaxis, :]

        value = np.full(testMET.shape[:2], -1.0) #Negative likelihood

        if self.enable_MET:
            value *= self.metTF(self.recoMET, testMET, self.covMET)
        
        if self.enable_mass:
            value *= self.massLikelihood(InvariantMass(testP4))

        if self.enable_mass_constraint:
            value *= self.mass_constraint(InvariantMass(testP4))
        
        if not self.enable_window: #default
            value[mask] = 0.000001
        else:
            value[~self.Window(InvariantMass(testP4))] = 1.0

        return value