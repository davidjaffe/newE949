#!/usr/bin/env python
'''
eventually combine E949/E787 with NA62 for Br(K+ => pi+,nu,nubar)
20191023
20191205 clean up. copy original to old_combiner.py and remove unused code
20200731 move to newE949 repo, more cleanup
'''
import math
import sys,os

import datetime
import numpy
import random
#import copy

import time

import re
#import glob # used in __init__

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,  AutoMinorLocator)
from scipy.stats import poisson

class combiner():
    def __init__(self,debug=0,drawEach=True,drawToFile=False,turnOnSyst=False,studyVar=False):

        
        self.debug      = debug
        self.drawEach   = drawEach
        self.drawToFile = drawToFile
        self.turnOnSyst = turnOnSyst
        self.studyVar   = studyVar
        print 'combiner.__init__ debug',self.debug,'drawEach',self.drawEach,'drawToFile',self.drawToFile,'turnOnSyst',self.turnOnSyst,'studyVar',self.studyVar

        # global variables controlled by turnOnSyst argument
        self.systOn = False
        self.systAcc = 0.10
        self.systN   = 100*5
        self.XforSysts = None
        self.vforSysts = None
        if self.systOn : print 'combiner.__init__ SYSTEMATIC VARIATIONS APPLIED. systAcc,systN',self.systAcc,self.systN

        # init of assumed branching fraction
        self.AssumedBr = self.AssumedBR = 1.73e-10 # PRD79, 092004 (2009) Table VIII
        print 'combiner.__init__ self.AssumedBr',self.AssumedBr,' *****************'

        self.Groups = None
        self.dataSets = None

        # define the binning to scan for the ratio wrt the assumed BR
        r = numpy.arange(0.,0.1,0.05)
        r = numpy.append( r, numpy.arange(0.1,0.8,0.05)  )
        r = numpy.append( r, numpy.arange(0.8,1.2,0.005) )
        r = numpy.append( r, numpy.arange(1.2,2.5,0.1)   )
        r = numpy.append( r, numpy.arange(2.5,10.,0.5)   )
        # smaller steps near minima of fits to subsets of data
        dx,step = 0.2,0.001
        abr = self.AssumedBr/1.e-10
        for x0 in [0.96/abr, 1.50/abr, 2.70/abr, 4.95/abr, 7.80/abr, 1.70/abr]:
            r = numpy.append( r, numpy.arange(x0-dx,x0+dx,step) )

        r = numpy.unique( numpy.sort(r) )
        self.ratioRange = r

        # define binning for CLs scans (only used by if CLs calculated)
        r = numpy.arange(0.,10.,0.5)
        r = numpy.arange(0.,3.,0.01)
        self.ratioRangeCLs = numpy.unique( numpy.sort(r) )
        print 'combiner.__init__ # bins in self.ratioRange',len(self.ratioRange),'self.ratioRangeCLs',len(self.ratioRangeCLs)

        # output directory for figures
        self.figDir = 'FIGURES/'
        
        print 'combiner.__init__ Did something'
        return
    def eclReader(self,dataset):
        '''
        translation of Joe/home/sher/test/ecl98inside.f
        if dataset is 'pnn1_E787_98' return list with [b_i,b_e] where b_i is bkgd in cell 
           containing 98C signal candidate, b_e = total bkg - b_i
        otherwise error exit
        '''
        if dataset=='pnn1_E787_98':
            boxbg98 = 0.0657
            isignal = 64 # cell with signal event (starting from 1)
            filename = 'DATA/bgtotal.inside.dat'
            f = open(filename,'r')
            n = None
            nRead = 0
            s,b = [],[]
            for line in f:
                words = line.split()
                if n is None:
                    n = int(words[0])
                else:
                    b.append( float(words[2]) )
                    s.append( float(words[3]) )
                    nRead += 1
            f.close()
            print 'combiner.eclReader',dataset,filename,
            if self.debug>0: 'n',n,'nRead',nRead,'raw b[:4]',b[:4],
            b = numpy.array(b) * boxbg98
            b_in_signal_cell = b[isignal-1]
            b_elsewhere = boxbg98 - b_in_signal_cell
            print 'total bkg {0:.4f} bkg in signal cell {1:.4f} bkg elsewhere {2:.4f}'.format(boxbg98,b_in_signal_cell,b_elsewhere)
            return [b_in_signal_cell, b_elsewhere]
        else:
            sys.exit('combiner.eclReader ERROR Unknown dataset '+dataset)
        return
    def reportSyst(self):
        '''
        return string with report on systematics parameters
        '''
        s = 'NO SYSTEMATIC VARIATIONS. Flag is False'
        if self.systOn: s = 'SYSTEMATIC VARIATION systAcc {0:.2f} systN {1}'.format(self.systAcc,self.systN)
        return s
    def studyVariations(self,cands,keyw='all', centralValue=1.):
        '''
        make a deepcopy of the input cands and monkey around
        to understand dependency of the BR that minimizes -2*loglike on various parameters
        '''
        import copy

        x = numpy.array(self.ratioRange)
        x = numpy.append(x, numpy.arange(0.9,1.1,.0001)*centralValue )
        x = numpy.sort(x)
        x = numpy.unique(x)
        ytitle = 'Br(K+ => pi+,nu,nubar)/'+str(self.AssumedBr)+ ' centralValue is {0:.2f}'.format(centralValue)

        if self.debug>0 : print 'combiner.studyVariations keyw',keyw,', cands.keys()',cands.keys()
        
    
        local_cands = self.collate(cands,keyw=keyw)
        if self.debug>0 : print 'combiner.studyVariations local_cands.keys()',local_cands.keys()


        vary = {'NK'    :numpy.arange( 0.90, 1.10, 0.01),
                'soverb':numpy.arange( 0.80, 1.20, 0.02) }

        for key in vary:
            br = []


            for v in vary[key]:
                title = 'Variation is ' + key + ' times ' + str(v)
                allc = copy.deepcopy(local_cands)
                CAND = allc[keyw]
                if type(CAND[key]) is list:
                    l = [v*z for z in CAND[key]]
                    CAND[key] = l
                else:
                    CAND[key] = v*CAND[key]
                allc = self.fillM2LL(allc,x)
                if self.debug>2: print 'combiner.studyVariations allc.keys()',allc.keys()
                if self.debug>2: print 'combiner.studyVariations allc[all].keys()',allc['all'].keys()
                m2ll = numpy.array(allc[keyw]['m2ll'])
                m2ll = m2ll-min(m2ll)
                if self.debug>2: print 'combiner.studyVariations',title,'len(x),len(m2ll)',len(x),len(m2ll)
                xatmin = x[numpy.argmin(m2ll)]
                if self.debug>1: print 'combiner.studyVariations',title,'minimized at',xatmin
                br.append( xatmin )
                #self.drawIt(x,m2ll,xtitle,ytitle,title,mark='-')
                del allc
            i = numpy.argmin(abs(numpy.array(br)-centralValue))
            i1= max(i-1,0)
            i2= min(i+1,len(br)-1)
            slope = (br[i2]-br[i1])/(vary[key][i2]-vary[key][i1])
            y1,y2= br[i1],br[i2]
            x1,x2= vary[key][i1],vary[key][i2]
            slope= (y2-y1)/(x2-x1)
            b    = (x2*y1 - y2*x1)/(x2-x1)
            best = (centralValue - b)/slope 
            print 'combiner.studyVariations {0} for {3} with centralValue {4:.2f}, slope of scale factor {1:.3f}, best scale factor {2:.3f}'.format(key,1./slope,best,keyw,centralValue)
            title = 'Variation of {0} for {2}. Best scale factor is {1:.3f} for centralValue {3:.2f}'.format(key,best,keyw,centralValue)
            self.drawIt(vary[key],br,key+' scale factor',ytitle,title,mark='o-')
            
        return
    def main(self):
        '''
        cleverly named main routine for loading E787/E949 data and computing -2*loglikelihood
        '''
        debug = self.debug
        drawEach = self.drawEach
        doCLs = False # if true, try to do CLs calculation
        x = numpy.array(self.ratioRange)
        if doCLs: xCLs = self.ratioRangeCLs
        xtitle = 'Br(K+ => pi+,nu,nubar)/'+str(self.AssumedBr)
        ytitle = '-2*loglikelihood'

        # load data, report it, scale it to have same assumed branching fraction, report that,
        # then, if requested, study fitted Br for variations in input parameters
        # then calculate m2ll = -2*loglike for each dataset
        cands = self.loadData()
        self.reportData(cands,mode='raw')
        cands = self.setAssBr(cands)
        self.reportData(cands,mode='same_assumed_Br')
        if self.studyVar :
            self.studyVariations(cands,keyw='all')
            self.studyVariations(cands,keyw='All pnn2',centralValue=5.11/1.73)
            self.studyVariations(cands,keyw='E949 pnn2',centralValue=7.89/1.73)
            
        # group candidates, calculate minimum of chi2, test function X, collect and report results by group
        # optionally draw -2loglike for each group
        groupCands = {}
        for group in sorted(self.Groups.keys()):
            groupCands[group] = self.collate(cands,keyw=group)[group]
            if debug>0: print 'combiner.main group',group,'groupCands[group].keys()',groupCands[group].keys(),'groupCands[group]',groupCands[group]
        if debug>0: print 'combine.main groupCands.keys()',groupCands.keys()
        groupCands = self.fillM2LL(groupCands)
        if doCLs: groupCands = self.fillX(groupCands,ratRange=xCLs)
        Results = {}
        gLL = {}
        if doCLs: gX  = {}
        for group in sorted(self.Groups.keys()):
            m2ll = numpy.array(groupCands[group]['m2ll'])
            gLL[group] = m2ll = m2ll-min(m2ll)
            if doCLs: gX[group] = groupCands[group]['X']
            xatmin = x[numpy.argmin(m2ll)]
            Results[group] = xatmin*self.AssumedBR
            if debug>0: print 'combine.main {0} minimized at BF {1:.2e}'.format(group,xatmin*self.AssumedBR)
            if drawEach :
                xtitle = 'Br(K->pi+nunubar)/{0:.2e}'.format(self.AssumedBR)
                ytitle = '-2*loglike'
                title  = '{0} minimized at {1:.2f} at BF {2:.2e}'.format(group,xatmin,xatmin*self.AssumedBR)
                xlims  = [ max(0.,xatmin-0.5),min(10.,xatmin+0.5)]
                ylims = [0.,0.5] 
                self.drawIt(x,m2ll,xtitle,ytitle,title,mark='o-',xlims=xlims,ylims=ylims,label=group)

        self.reportGroups(Results)
        title = 'Groups'
        loc = 'best'

        if doCLs: 
            self.getCLs(groupCands,group='all')
            self.drawMany(xCLs,gX,xtitle,gX.keys(),title,loc=loc)

        # draw -2LL vs Br/nominal for all groups on diffferent scales
        self.drawMany(x,gLL,xtitle,gLL.keys(),title,loc=loc)
        self.drawMany(x,gLL,xtitle,gLL.keys(),title+' restricted x and y ranges',ylims=[0.,4.],xlims=[0.,2.],loc=loc)
        self.drawMany(x,gLL,xtitle,gLL.keys(),title+' fanatical x and y ranges',ylims=[0.,0.2],xlims=[0.8,1.2],loc=loc)
        
        M2LL = {}
        # systematics study : combined candidates and likelihood
        if self.turnOnSyst:
            self.systOn = True
            if self.systOn : print 'combiner.main Now produce combined loglikelihood with systematics. ',self.reportSyst()
            allcands = self.collate(cands)
            allcands = self.fillM2LL(allcands)
            m2ll = numpy.array(allcands['all']['m2ll'])
            m2ll = m2ll-min(m2ll)
            xatmin = x[numpy.argmin(m2ll)]
            Bratmin = xatmin*self.AssumedBr
            if debug>0: print 'combiner.main allcands minimized at',xatmin
            title = '-2*loglikelihood with systematics included'
            label = 'Minimum at {0:.3f} BF={1:.3e} systN={2}'.format(xatmin,Bratmin,self.systN)
            print 'combiner.main',title,label
            self.drawIt(x,m2ll,xtitle,ytitle,title,mark='-',label=label)
            self.drawIt(x,m2ll,xtitle,ytitle,title+' restrict ranges',mark='-',xlims=[0.5,1.5],ylims=[0.,0.1],label=label)
            M2LL['all_with_syst'] = m2ll

            self.systOn = False

        return
    def m2loglike(self,cand,RATIO):
        '''
        calculate -2 * log likelihood from NK,Atot,[s/b], given ratio = BR/self.AssumedBr
        optionally include averaging over systematic variation of global acceptance 
        '''
        if type(cand['NK']) is list:
            NKlist = cand['NK']
            Atotlist = cand['Atot']
        else:
            NKlist = [cand['NK']]
            Atotlist = [cand['Atot']]
        soverb = cand['soverb']

        v,X = [1.],[1.]
        if self.systOn:
            if self.XforSysts is None:
                slo,shi=-5.,5.
                ds = (shi-slo)/float(self.systN)
                X = numpy.arange(slo,shi,ds)*self.systAcc + 1.
                import scipy.stats
                v = scipy.stats.norm.pdf(X,1.,self.systAcc)
                self.XforSysts = X
                self.vforSysts = v
            else:
                X = self.XforSysts
                v = self.vforSysts

        like = 0.
        totwt= 0.
        for f,wt in zip(X,v):
            ratio = f*RATIO
            totwt += wt
            for NK,Atot in zip(NKlist,Atotlist):
                like += ratio*self.AssumedBr*NK*Atot
            for x in soverb:
                like -= math.log(1. + ratio*x)
        like = like/totwt
        like *= 2.
        return like
    def testFcn(self,cand,RATIO):
        S = RATIO * cand['sigi']
        return self.testX(S,cand['bkgi'],cand['candi'])
    def testX(self,S,B,D):
        '''
        return X = prod_i X_i where X_i = pois(d_i,s_i+b_i)/pois(d_i,b_i) for pois(k,mu) = exp(-mu)*mu^k/k!
        '''
        X = 1.
        for s,b,d in zip(S,B,D):
            num,den = self.poisProbs(s,b,d)
            X *= num/den
        return X
    def poisProbs(self,s,b,d):
        '''
        return poisson probability for s+b and b only
        '''
        sbProb = poisson.pmf(d,s+b)
        bProb  = poisson.pmf(d,b)
        return sbProb,bProb
    def getCLs(self,groupCands,group='all'):
        '''
        perform calculations needed for CLs estimates
        '''
        groupList = sorted(groupCands.keys())
        if group is not None : groupList = [group]
        maxc = 3 # 5 # maximum number of candidates to consider per cell
        crange = range(maxc+1)
        for gkey in groupList:
            Cand = groupCands[gkey]
            bkgi = Cand['bkgi']
            sigi = Cand['sigi']
            candi= Cand['candi']
            Xobs = self.testX(sigi,bkgi,candi)
            bProbs,sbProbs,Xstats = [],[],[]
            for i,cell in enumerate(candi):
                b = bkgi[i]
                s = sigi[i]
                for d in crange:
                    sbP,bP = self.poisProbs(s,b,d)
                    sbProbs.append(sbP)
                    bProbs.append(bP)
                    Xstats.append( sbP/bP )
            bProbs = self.reshape( bProbs, len(candi),len(crange))
            sbProbs= self.reshape(sbProbs, len(candi),len(crange))
            Xstats = self.reshape( Xstats, len(candi),len(crange))
            ## here is little test to check if calculation of Xstats gives test statistic X=Xobs
            ## when Xstats is indexed with cell# and # of observed candidates
            Xcheck = 1.
            for i,d in enumerate(candi):
                X = Xstats[i,d]
                Xcheck *= X
            print 'combiner.getCLs group',group,'#cells',len(candi),'Xobs',Xobs,'Xcheck',Xcheck

        elapsedTime = {}
        time1 = time.time()
        print 'combiner.getCLs create list of',len(candi),'combinations of up to',maxc,'candidates/cell'
        Lcomb = self.getCombos(maxc, len(candi))
        time2 = time.time()
        elapsedTime['create list of combos']= time2-time1
        print elapsedTime
        print 'combiner.getCLs Now calculate X,sbProb,bProb for all combinations'
        Xcomb,Psb,Pb = [],[],[]
        for combo in Lcomb:
            X,psb,pb = 1.,1.,1.
            for i,d in enumerate(combo):
                X *= Xstats[i,d]
                psb *= sbProbs[i,d]
                pb  *= bProbs[i,d]
            Xcomb.append(X)
            Psb.append(psb)
            Pb.append(pb)
        time3 = time.time()
        elapsedTime['calculate X,sbProb,bProb'] = time3-time2
        print elapsedTime
        print 'combiner.getCLs Sort X,sbProb,bProb'
        Xcomb = numpy.array(Xcomb)
        Psb   = numpy.array(Psb)
        Pb    = numpy.array(Pb)
        indX  = Xcomb.argsort()
        X_sorted  = Xcomb[indX]
        Psb_sorted= Psb[indX]
        Pb_sorted = Pb[indX]
        time4 = time.time()
        elapsedTime['sort X,sbProb,bProb'] = time4-time3
        print elapsedTime
        return
    def getCombos(self,maxc,nways):
        '''
        get list of combinations of up to maxc candidates/cell taken nways ways
        Avoid generating list if it is already available in a file
        '''
        import itertools, pickle

        filename = 'combos_'+str(maxc)+'_'+str(nways)+'.pickle'
        if os.path.isfile(filename):
            f = open(filename,'r')
            Lcomb = pickle.load(filename)
            f.close()
            print 'combiner.getCombos Read combos from',filename
        else:
            Lcomb = list(itertools.product(range(maxc+1),repeat=nways))
            f = open(filename,'wb')
            pickle.dump(Lcomb,f)
            f.close()
            print 'combiner.getCombos Write combos to',filename
        return Lcomb
    def reshape(self,A,n1,n2):
        '''
        return container A after conversion numpy array and reshaping as (n1,n2)
        '''
        if type(A) is list : A = numpy.array( A )
        A.shape = (n1,n2)
        return A
    def fillX(self,cands,ratRange=None):
        '''
        loop over datasets add array of test function X for ratio in ratRange to dict cands
        Note input dict cands is modified by this routine
        '''
        ratioRange = ratRange
        if ratioRange is None : ratioRange = self.ratioRange
        for dataset in sorted(cands):
            if self.debug>0 : print 'combiner.fillX Process dataset',dataset
            cand = cands[dataset]
            if 'X' in cand: sys.exit('combiner.fillX ERROR key `X` already exists for dataset '+dataset+', due to multi-calls to this routine?')
            X = []
            for ratio in ratioRange:
                x = self.testFcn(cand,ratio)
                X.append( x )
            X = numpy.array( X )
            cands[dataset]['X'] = X
        return cands
    def fillM2LL(self,cands,ratRange=None):
        '''
        loop over datasets and add array of -2*loglike(ratio) for ratio in ratRange to dict cands
        Note that input dict cands is modified by this module.
        ratRange defaults to self.ratioRange if no input is provided
        '''
        debug = self.debug
        ratioRange = ratRange
        if ratRange is None : ratioRange = self.ratioRange
        if debug>0 : print 'combiner.fillM2LL cands',cands
        for dataset in sorted(cands):
            cand = cands[dataset]
            if debug>0: print 'combiner.fillM2LL dataset,cand',dataset,cand
            if 'm2ll' in cand:
                sys.exit('combiner.fillM2LL ERROR key `m2ll` already exists for dataset '+dataset+', perhaps due to multiple calls to this routine?')
            m2ll = []
            for ratio in ratioRange:
                x = self.m2loglike(cand,ratio)
                m2ll.append(x)
            cands[dataset]['m2ll'] = m2ll
        return cands
    def collate(self,cands,keyw='all'):
        '''
        create a dict with candidates from datasets specified by keyw
        AssumedBr must be the same for all combined datasets
        Cannot perform collation if a dataset in cands contains -2*loglike array.
        combination defined by keyw must already be defined

        '''
        allcands = {}
        allcands[keyw] = {'NK':[], 'Atot':[], 'soverb':[], 'AssumedBr':None, 'Btot':[], 'sigi':[], 'bkgi':[], 'candi':[]}
        groups = self.Groups
        AssBr = None

        if keyw=='all':
            setList = sorted(cands.keys())
        elif keyw in groups:
            setList = groups[keyw]
        else:
            sys.exit('combiner.collate ERROR Invalid keyw '+keyw)

        
        for dataset in setList: 
            if 'm2ll' in cands[dataset]:
                sys.exit('combiner.collate ERROR key `m2ll` in input dict cands for dataset '+dataset)
            cand = cands[dataset]
            NK = cand['NK']
            Atot = cand['Atot']
            Btot = cand['Btot']
            soverb = cand['soverb']
            sigi   = cand['sigi']
            bkgi   = cand['bkgi']
            candi  = cand['candi']
            if AssBr is None: AssBr = cand['AssumedBr']
            if AssBr!=cand['AssumedBr']:
                print 'combiner.collate ERROR dataset,AssumedBr',dataset,cand['AssumedBr'],'is not equal to',AssBr,'found for first dataset'
                sys.exit('combiner.collate ERROR inconsistent assumed Br')
            allcands[keyw]['NK'].append( NK )
            allcands[keyw]['Atot'].append( Atot )
            allcands[keyw]['Btot'].append( Btot )
            allcands[keyw]['soverb'].extend( soverb )
            allcands[keyw]['sigi'].extend( sigi )
            allcands[keyw]['bkgi'].extend( bkgi )
            allcands[keyw]['candi'].extend( candi )
            allcands[keyw]['AssumedBr'] = AssBr
        # change lists to numpy arrays for later use
        for k in ['sigi','bkgi','candi']: allcands[keyw][k] = numpy.array( allcands[keyw][k] )
        return allcands
    def loadData(self):
        '''
        return dict loaded with all E787/E949 data
        For each dataset we have 
        dataset name
        journal reference
        NK = stopped kaons
        Atot = total acceptance
        Ncand = number of candidates
        AssumedBr = assumed B(K+ => pi+,nu,nubar) used for s_i/b_i ratio
        soverb = s_i/b_i = signal to background ratio in ith cell (cells containing candidates)
        Btot = total background
        Ctot = total number of cells for CLs
        bkgi = bkgd in ith cell, final cell contains no candidates if Ctot>Ncand
        '''
        cands = {}

        self.dataSets = []
        
        dataset = 'pnn1_E787_95-7'
        self.dataSets.append( dataset )
        journal = 'PRL88_041803_and_PRL101_191802'
        NK = 3.2e12
        Atot = 2.1e-3
        Btot = 0.08 ## +- 0.03
        Ctot = 2
        Ncand = 1
        soverb = [59.] ## PRL101
        b = 0.00646 ## tight golden region, Bergbusch thesis Table 4.21
        bkgi   = [b, Btot-b] ## 
        AssumedBr = 1.73e-10 ## PRL101
        sigi,candi = self.fillSig(Atot,Ctot,Ncand,NK,AssumedBr,soverb,bkgi)
        cands[dataset] = {'NK':NK, 'Atot':Atot, 'Btot':Btot, 'Ctot':Ctot, 'bkgi':bkgi, 'sigi':sigi, 'candi':candi, 'Ncand':Ncand, 'soverb':soverb, 'AssumedBr':AssumedBr, 'journal':journal}

        dataset = 'pnn1_E787_98'
        self.dataSets.append( dataset )
        journal = 'PRL88_041803_and_PRL101_191802'
        NK = 2.7e12
        Atot = 1.96e-3
        Btot = 0.066 ## +0.044-0.025
        Ctot = 2 
        Ncand = 1
        soverb = [8.2]  ## PRL 101
        bkgi   = self.eclReader(dataset)
        AssumedBr = 1.73e-10 ## PRL101
        sigi,candi = self.fillSig(Atot,Ctot,Ncand,NK,AssumedBr,soverb,bkgi)
        cands[dataset] = {'NK':NK, 'Atot':Atot, 'Btot':Btot, 'Ctot':Ctot, 'bkgi':bkgi, 'sigi':sigi, 'candi':candi, 'Ncand':Ncand, 'soverb':soverb, 'AssumedBr':AssumedBr, 'journal':journal}

        dataset = 'pnn1_E949'
        self.dataSets.append( dataset )
        journal = 'PRD77_052003_and_PRL101_191802'
        NK = 1.77e12
        Atot = 2.22e-3  ## +- 0.07 +- 0.15 e-3
        Btot = 0.30 ## +-0.03 for so-called extended signal region
        Ctot = 2
        AssumedBr = self.AssumedBr
        Ncand = 1
        b = 5.7e-5
        s = 3.628e5*AssumedBr
        bkgi = [b, Btot-b]
        soverb = [1.1] ## PRL101 (same as PRD77)
        sigi,candi = self.fillSig(Atot,Ctot,Ncand,NK,AssumedBr,soverb,bkgi)
        cands[dataset] = {'NK':NK, 'Atot':Atot, 'Btot':Btot, 'Ctot':Ctot, 'bkgi':bkgi, 'sigi':sigi, 'candi':candi, 'Ncand':Ncand, 'soverb':soverb, 'AssumedBr':AssumedBr, 'journal':journal}

        dataset = 'pnn2_E787_96'
        self.dataSets.append( dataset )
        journal = 'PLB537_2002_211'
        NK = 1.12e12
        Atot = 0.765e-3
        Btot = 0.734 ## +-0.117
        Ctot = 1 ## only one cell defined
        Ncand = 1
        b = Btot
        bkgi = [b]        
        AssumedBr = self.AssumedBr
        s = NK*Atot*AssumedBr*float(Ncand)
        soverb = [s/b]
        sigi,candi = self.fillSig(Atot,Ctot,Ncand,NK,AssumedBr,soverb,bkgi)
        cands[dataset] = {'NK':NK, 'Atot':Atot, 'Btot':Btot, 'Ctot':Ctot, 'bkgi':bkgi, 'sigi':sigi, 'candi':candi, 'Ncand':Ncand, 'soverb':soverb, 'AssumedBr':AssumedBr, 'journal':journal}

        dataset = 'pnn2_E787_97'
        self.dataSets.append( dataset )
        journal = 'PRD70_037102'
        NK = 0.61e12
        Atot = 0.97e-3
        Btot = 0.49 ## +-0.16
        Ctot = 1 ## only one cell defined
        bkgi = [Btot] 
        Ncand = 0
        AssumedBr = self.AssumedBr
        soverb = []
        sigi,candi = self.fillSig(Atot,Ctot,Ncand,NK,AssumedBr,soverb,bkgi)
        cands[dataset] = {'NK':NK, 'Atot':Atot, 'Btot':Btot, 'Ctot':Ctot, 'bkgi':bkgi, 'sigi':sigi, 'candi':candi, 'Ncand':Ncand, 'soverb':soverb, 'AssumedBr':AssumedBr, 'journal':journal}

        dataset = 'pnn2_E949'
        self.dataSets.append( dataset )
        journal = 'PRD79_092004'
        NK = 1.71e12
        Atot = 1.37e-3 #(+-0.14e-3)
        Btot = 0.93 ## +- 0.17 +0.32-0.24
        Ctot = 4
        Ncand = 3
        AssumedBr = 1.73e-10
        soverb = [0.47, 0.42, 0.20]
        bkgi   = [0.243, 0.027, 0.379]
        bkgi.append(Btot - sum(bkgi))
        sigi,candi = self.fillSig(Atot,Ctot,Ncand,NK,AssumedBr,soverb,bkgi)
        cands[dataset] = {'NK':NK, 'Atot':Atot, 'Btot':Btot, 'Ctot':Ctot, 'bkgi':bkgi, 'sigi':sigi, 'candi':candi, 'Ncand':Ncand, 'soverb':soverb, 'AssumedBr':AssumedBr, 'journal':journal}

        groups = {'pnn1_pub': ['pnn1_E787_95-7','pnn1_E787_98','pnn1_E949'],
                    'All E787': ['pnn1_E787_95-7','pnn1_E787_98','pnn2_E787_96','pnn2_E787_97'],
                    'All E949': ['pnn1_E949', 'pnn2_E949'],
                    'All pnn1': ['pnn1_E787_95-7','pnn1_E787_98','pnn1_E949'],
                    'All pnn2': ['pnn2_E787_96','pnn2_E787_97','pnn2_E949'],
                    'all'     : self.dataSets,
                    'E949 pnn1': ['pnn1_E949'],
                    'E949 pnn2': ['pnn2_E949']
                      }
        self.Groups = groups
        # E949 Technote K074.v1 Table 89 for fitted BR in 1.e-10 units
        # labelled 'Joss' in this file for reasons that cannot be revealed
        self.Joss  = {'pnn1_pub': 1.47, 
                    'All E787'  : 1.49, 
                    'All E949'  : 2.80, 
                    'All pnn1'  : 1.46,
                    'All pnn2'  : 5.11, 
                    'all'       : 1.73, 
                    'E949 pnn1' : 0.96,
                    'E949 pnn2' : 7.89
                      }

        return cands
    def fillSig(self,Atot,Ctot,Ncand,NK,AssumedBr,soverb,bkgi):
        ''' 
        return lists sigi, candi given Atot,Ctot,Ncand,NK,AssumedBr,soverb,bkgi
        where sigi = signal in ith cell given AssumedBr
        and candi  = number of candidates in ith cell
        ordering in sigi,candi, should match bkgi
        Atot = total acceptance
        Ctot = total number of cells
        Ncand= total number of candidates
        NK   = number of stopped kaons
        AssumedBr = BF assumed for soverb
        soverb = list of signal_i/bkg_i for i = 1,Ncand
        bkgi   = background in ith cell
        '''
        candi, sigi = None, None
        if Ncand==0 :
            candi = [0 for x in range(Ctot)]
            s = NK*AssumedBr*Atot
            sigi = [s]
        else:
            candi = [1 for x in soverb]
            candi.extend( [0 for x in range(Ctot-len(soverb))] )
            sigi = []
            Ae   = Atot
            for i,sb in enumerate(soverb):
                b = bkgi[i]
                s = b*sb
                sigi.append(s)
                Aj= s/NK/AssumedBr   ## acceptance of this cell
                Ae -= Aj             ## remaining acceptance (Ae = Acceptance elsewhere)
            s = NK*AssumedBr*Ae
            if s>1.e-10 : sigi.append(s)  ## avoid adding remaining acceptance that is consistent with zero within precision
        return sigi,candi
    def reportGroups(self,Results):
        '''
        report content of self.Groups with fitted BR and Joss's fitted BR
        '''
        if self.Groups is None: sys.exit('combiner.reportGroups ERROR self.Groups not initialized')
        print '\nColumn labels: BF=Branching Fraction that minimizes 2loglike(group), K074=BF from technote K074, BF/K = ratio BF and K074'
        print '{0:^12} {1:^12} {2:^6} {3:<15} | {4}'.format('BF(1e-10)','K074(1e-10)','BF/K','Group name','data sets')
        for group in sorted(self.Groups.keys()):
            mine = Results[group]/1.e-10
            Joss = self.Joss[group]
            r    = mine/Joss
            print '{0:>12.2f} {1:<12.2f} {2:^6.2f} {3:<15} |'.format(mine,Joss,r,group),'%s' % ' '.join(map(str,self.Groups[group]))
        return
            
    def setAssBr(self,cands):
        '''
        return cands with s/b using a common assumed branching fractions self.AssumedBr
        '''
        for dataset in cands:
            cand = cands[dataset]
            factor = self.AssumedBr/cand['AssumedBr']
            soverb = cand['soverb']
            newsb = [factor*x for x in soverb]
            cand['soverb'] = newsb
            cand['AssumedBr'] = self.AssumedBr
        print 'combiner.setAssBr set assumed Br to',self.AssumedBr,'for s/b'
        return cands
    def reportData(self,cands,mode=None):
        '''
        report contents of all data in cands
        mode = 'raw' ==> no change to data
        mode = 'sameBr' ==> give all s/b at self.AssumedBr
        '''
        units = {'NK':1e12, 'Atot':1e-3, 'AssumedBr':1e-10, 'SES':1.e-10}

        
        print 'combiner.reportData mode is',mode,'. `AssBr` means Assumed Branching fraction of K+ => pi,nu,nubar for sig/bkg column. Btot=total bkgd, Ctot=total cells for CLs'
        print '{0:<15} {1:>5}({2:5.0e}) {3:>5}({4:5.0e}) {5:>5}({6:5.0e}) {7:>5}({8:5.0e}) {9:>5} {10:>15} {11:>5} {12:>5} {13:^22} {14:^22} {15:5} {16:<45}'.format('dataset','NK',units['NK'],'Atot',units['Atot'],'SES',units['SES'],'AssBr',units['AssumedBr'],'Ncand','sig/bkg','Btot','Ctot','background/cell','signal/cell','Candi','journal')
        #           0      1    2           3       4            5      6            7      8                 9        10        11
        for dataset in sorted(cands):
            cand = cands[dataset]
            NK = cand['NK']/units['NK']
            Atot = cand['Atot']/units['Atot']
            SES = 1./cand['NK']/cand['Atot']/units['SES']
            AssBr = cand['AssumedBr']/units['AssumedBr']
            Ncand = cand['Ncand']
            soverb= cand['soverb']
            journal = cand['journal']
            Btot = cand['Btot']
            Ctot = cand['Ctot']
            bpercell = cand['bkgi']
            sigi  = cand['sigi']
            candi = cand['candi']
            wsb,wbc,ws,wc = '','','',''
            for x in soverb: wsb += '{0:>5.2f}'.format(x)
            for x in bpercell: wbc += '{0:>.2g} '.format(x)
            for x in sigi: ws += '{0:>.2g} '.format(x)
            for x in candi: wc += '{0:1}'.format(x)
            
            print '{0:<15} {1:>12.2f} {2:>12.2f} {3:>12.2f} {4:>12.2f} {5:>5} {6:>15} {7:>5.2f} {8:>5} {9:>22} {10:>22} {11:5} {12:<45}'.format(dataset,NK,Atot,SES,AssBr,Ncand,wsb,Btot,Ctot,wbc,ws,wc,journal)
            
        return
    def titleAsFilename(self,title):
        '''
        return ascii suitable for use as a filename
        list of characters to be replaced is taken from https://stackoverflow.com/questions/4814040/allowed-characters-in-filename
        '''
        r = {'_': [' ', ',',  '\\', '/', ':', '"', '<', '>', '|'], 'x': ['*']}
        filename = title
        filename = ' '.join(filename.split()) # substitutes single whitespace for multiple whitespace
        for new in r:
            for old in r[new]:
                if old in filename : filename = filename.replace(old,new)
        return filename
    def drawIt(self,x,y,xtitle,ytitle,title,ylog=False,xlims=None,ylims=None,mark='o-',label=''):
        '''
        draw graph defined by x,y

        '''
        plt.clf()
        plt.grid()
        plt.title(title)

        X = numpy.array(x)
        Y = numpy.array(y)
        plt.plot(X,Y,mark,label=label)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        if ylog : plt.yscale('log')
        if xlims is not None: plt.xlim(xlims)
        if ylims is not None: plt.ylim(ylims)

        plt.legend(loc='best')

        if self.drawToFile:
            fn = self.titleAsFilename(title)
            figpdf = 'FIG_'+fn + '.pdf'
            figpdf = self.figDir + figpdf
            plt.savefig(figpdf)
            print 'combiner.drawIt wrote',figpdf
        else:
            plt.show()
        return    
    def drawMany(self,x,y,xtitle,ytitle,title,ylog=False,xlims=None,ylims=None,loc='best'):
        '''
        draw many graphs with same abscissa and different ordinate values on same plot defined by x,y
        y = dict
        ytitle = keys of dict

        '''
        fig,ax = plt.subplots()
        plt.grid()
        plt.title(title)
        major = 1.
        if xlims is not None:
            if xlims[1]-xlims[0]<=2: major = (xlims[1]-xlims[0])/10
        ax.xaxis.set_major_locator(MultipleLocator(major))
        minor = major/5.
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
        if self.debug>1: print 'combiner.drawMany major,minor',major,minor,'xlims',xlims


        ls = ['-','--','-.',':','-','--',':']
        ls.extend(ls[::-1])
        c  = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        c.extend(c)
        
        X = numpy.array(x)
        for i,key in enumerate(ytitle):
            Y = numpy.array(y[key])
            ax.plot(X,Y,linestyle=ls[i],color=c[i],label=key)

        plt.xlabel(xtitle)
#        plt.ylabel(ytitle)
        if ylog : ax.yscale('log')
        if xlims is not None: plt.xlim(xlims)
        if ylims is not None: plt.ylim(ylims)

            
        ax.legend(loc=loc)
        if self.drawToFile : 
            fn = self.titleAsFilename(title)
            figpdf = 'FIG_'+fn + '.pdf'
            figpdf = self.figDir + figpdf
            plt.savefig(figpdf)
            print 'combiner.drawMany wrote',figpdf
        else:
            plt.show()
        return    
if __name__ == '__main__' :

    drawEach = False # draw loglikelihood for each dataset?
    debug    = 0 # >0 gives output
    drawToFile = False # plots go to file instead of to terminal (use savefig() instead of show())
    turnOnSyst = False # include systematics in BR determination
    studyVar   = False # variation of inputs for alternate BR determinations
    if len(sys.argv)>1:
        if sys.argv[1].lower()=='draweach': drawEach=True
        if sys.argv[1].lower()=='help' : sys.exit( 'usage: python combiner.py drawEach debug drawToFile turnOnSyst studyVar' )
    if len(sys.argv)>2:
        try: 
            debug = int(sys.argv[2])
        except ValueError:
            debug = 0
    if len(sys.argv)>3:
        if sys.argv[3].lower()=='drawtofile' : drawToFile = True
    if len(sys.argv)>4:
        if sys.argv[4].lower()=='turnonsyst' : turnOnSyst = True
    if len(sys.argv)>5:
        if sys.argv[5].lower()=='studyvar'   : studyVar   = True
        
    cb = combiner(debug=debug,drawEach=drawEach,drawToFile=drawToFile,turnOnSyst=turnOnSyst,studyVar=studyVar)
    cb.main()
