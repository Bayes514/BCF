/* Open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 **
 ** This program is free software: you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation, either version 3 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 ** GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program. If not, see <http://www.gnu.org/licenses/>.
 **
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */
#include "ao_tan.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>

 BCF::BCF() :
trainingIsFinished_(false)
{
}

BCF::BCF(char* const *&, char* const *) :
xxyDist_(), trainingIsFinished_(false)
{
    name_ = "BCF";
    //printf("AOTAN Gen: HXXY 版本 Local: HxxY 版本\n");
}

BCF::~BCF(void)
{
}

void BCF::reset(InstanceStream &is)
{
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    trainingIsFinished_ = false;

    //safeAlloc(parents, noCatAtts_);
    gen_parents_.resize(noCatAtts_);
    for(int x = 0;x<this->noCatAtts_;x++){
        gen_parents_[x].resize(noCatAtts);
        for (CategoricalAttribute a = 0; a < noCatAtts; a++)
        {
            gen_parents_[x].clear(); //？？
        }
    }

    IXP_Dict.resize(this->noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        IXP_Dict[a].resize(this->noCatAtts_);
        for (CategoricalAttribute b = 0; b < noCatAtts_; b++)
        {
            IXP_Dict[a][b] = 0;
        }

    }




    xxyDist_.reset(is);
}

void BCF::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void BCF::initialisePass()
{
    assert(trainingIsFinished_ == false);
    //learner::initialisePass (pass_);
    //	dist->clear();
    //	for (CategoricalAttribute a = 0; a < meta->noCatAtts; a++) {
    //		parents_[a] = NOPARENT;
    //	}
}

void BCF::train(const instance &inst)
{
    xxyDist_.update(inst);
}

void BCF::classify_local(std::vector<CategoricalAttribute> & parents_,const instance &inst, std::vector<double> &classDist)
{
    //printf("classify_local\n");
    for (CatValue y = 0; y < noClasses_; y++)
    {
        classDist[y] = xxyDist_.xyCounts.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        const CategoricalAttribute parent = parents_[x1];

        if (parent == NOPARENT)
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
            }
        } else
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
                        inst.getCatVal(parent), y);
            }
        }
    }

    normalise(classDist);
}

double BCF::H(const instance &inst, std::vector<unsigned int> &parrent){

    double p_sum =0;
    std::vector<double> classDist_local;
    classDist_local.resize(this->noClasses_);
    for (CatValue y = 0; y < noClasses_; y++)
    {
        classDist_local[y] = xxyDist_.xyCounts.p(y) * 100000;
    }


    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        const CategoricalAttribute parent = parrent[x1];

        if (parent == NOPARENT)
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist_local[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
            }
        }
        else
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist_local[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
                        inst.getCatVal(parent), y);
            }
        }
    }


    normalise(classDist_local);
    for (CatValue y = 0; y < noClasses_; y++)
    {
        double p = classDist_local[y];
        p_sum-= (log(p));
    }
    return p_sum;

}

double BCF::IXP(int x, std::vector<unsigned int> &parrent)
{

    double p_sum=0;
    const CategoricalAttribute parent = parrent[x];
    const double totalCount = xxyDist_.xyCounts.count;

    if(parent == this->NOPARENT)
    {
        return 0.0;
    }

    if(IXP_Dict[x][parent] != 0){
        return IXP_Dict[x][parent];
    }


    for (unsigned int xv = 0; xv < this->instanceStream_->getNoValues(x); xv++)
    {
        for (unsigned int xvp = 0; xvp < this->instanceStream_->getNoValues(parent); xvp++)
        {
            const InstanceCount avpvCount = xxyDist_.getCount(x, xv, parent, xvp);
            if (avpvCount)
            {
                p_sum += (avpvCount / totalCount) * log2(avpvCount / ((xxyDist_.xyCounts.getCount(x, xv) / totalCount)
                            * xxyDist_.xyCounts.getCount(parent, xvp)));
            }


        }


    }
    IXP_Dict[x][parent] = p_sum;
    return p_sum;
}



void BCF::softmax(std::vector<double> &data){
    double sum = 0;
    for(int i = 0;i< data.size(); i++){
        sum += exp(data[i]);
    }
    for(int i = 0;i< data.size(); i++){
        //printf("%.3f -> ", data[i]);
        data[i] = exp(data[i]) / sum;
        //printf("%.3f\n", data[i]);

    }

}



void BCF::classify(const instance &inst, std::vector<double> &classDist)
{
    //分类
    std::vector<std::vector<double> >Gen;


    for(int attr = 0;attr<this->noCatAtts_;attr++){
        std::vector<double>gen_res = std::vector<double>(this->noClasses_,0);
        classify_local(gen_parents_[attr],inst,gen_res);
        normalise(gen_res);
        Gen.push_back(gen_res);
    }


    for(int y = 0; y< this->noClasses_; y++){
        double res = 0;
        for(int attr = 0; attr< this->noCatAtts_; attr++){
            res+= 1.0/Gen[attr][y];
        }
        classDist[y] = 1.0/res;
    }

    normalise(classDist);
}

void BCF::finalisePass()
{
    assert(trainingIsFinished_ == false);

    // 第一个下标是 子节点   第二个是父节点
    // // H(Xj | Xi, Y)
    std::vector<std::vector<double> > HXXY = std::vector<std::vector<double> >(this->noCatAtts_,std::vector<double>(this->noCatAtts_,0));
    for(int x1 = 0;x1<this->noCatAtts_;x1++){
        for(int x2 = 0;x2<this->noCatAtts_;x2++){
            if(x1 == x2){
                continue;
            }
            double hxxy =0;
            for(int x1v = 0;x1v<this->instanceStream_->getNoValues(x1);x1v++){
                for(int x2v = 0;x2v<this->instanceStream_->getNoValues(x2);x2v++){
                    for(int y = 0;y<this->noClasses_;y++){
                        hxxy += this->xxyDist_.jointP(x1,x1v,x2,x2v,y) * log2(this->xxyDist_.p(x1,x1v,x2,x2v,y));
                    }
                }
            }
            hxxy = -hxxy;
            HXXY[x1][x2] = hxxy;
        }
    }

    for(int attr = 0;attr <this->noCatAtts_;attr++){
        CategoricalAttribute firstAtt = attr;
        this->gen_parents_[attr][firstAtt] = NOPARENT;

        float *maxWeight;
        CategoricalAttribute *bestSoFar;
        CategoricalAttribute topCandidate = firstAtt;
        std::set<CategoricalAttribute> available;

        safeAlloc(maxWeight, noCatAtts_);
        safeAlloc(bestSoFar, noCatAtts_);

        maxWeight[firstAtt] = std::numeric_limits<float>::max();

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
        {
            if(a == firstAtt){
                continue;
            }
            maxWeight[a] = HXXY[a][firstAtt];
            if (HXXY[a][firstAtt] < maxWeight[topCandidate])
                topCandidate = a;
            bestSoFar[a] = firstAtt;
            available.insert(a);
       }

       while (!available.empty())
       {
            const CategoricalAttribute current = topCandidate;
            gen_parents_[attr][current] = bestSoFar[current];
            available.erase(current);

            if (!available.empty())
            {
                topCandidate = *available.begin();
                for (std::set<CategoricalAttribute>::const_iterator it =
                        available.begin(); it != available.end(); it++)
                {
                    if (maxWeight[*it] > HXXY[*it][current])
                    {
                        maxWeight[*it] = HXXY[*it][current];
                        bestSoFar[*it] = current;
                    }

                    if (maxWeight[*it] < maxWeight[topCandidate])
                        topCandidate = *it;
                }
            }
        }

        delete[] bestSoFar;
        delete[] maxWeight;


    }





    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool BCF::trainingIsFinished()
{
    return trainingIsFinished_;
}
