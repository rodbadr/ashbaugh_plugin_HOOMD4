// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_ASHBAUGH_H__
#define __PAIR_EVALUATOR_ASHBAUGH_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairAshbaugh.h
    \brief Defines the pair evaluator class for the example potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {

class EvaluatorPairAshbaugh
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar lj1;     //!< energy LJ times 4
        Scalar lj2;      //!< particle diameter LJ to power 6
	    Scalar lam;     //!< lambda parameter
        Scalar rminsq;
        Scalar e_shift;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

#ifndef __HIPCC__
	    param_type() : lj1(0), lj2(0), lam(0), rminsq(0), e_shift(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            auto sigma(v["sigma"].cast<Scalar>());
            auto epsilon(v["epsilon"].cast<Scalar>());
            lam = v["lam"].cast<Scalar>();
            
            Scalar sigma_6 = sigma * sigma * sigma * sigma * sigma * sigma;
            lj1 = Scalar(4.0) * epsilon * sigma_6 * sigma_6;
            lj2 = Scalar(4.0) * epsilon * sigma_6;
            rminsq = 1.2599210499 * sigma * sigma;
            e_shift = epsilon * (Scalar(1.0) - lam);
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["epsilon"] = e_shift / (1.0 - lam);
            v["sigma"] = pow(lj2 / 4.0 / (e_shift / (1.0 - lam)), 1. / 6.);
	        v["lam"] = lam;
            return v;
            }
#endif
        }
#ifdef SINGLE_PRECISION
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairAshbaugh(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
      : rsq(_rsq), rcutsq(_rcutsq), lj1(_params.lj1), lj2(_params.lj2), lam(_params.lam),
        rminsq(_params.rminsq), e_shift(_params.e_shift)
        {
        }

    //! Example doesn't use diameter
    DEVICE static bool needsDiameter()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj) { }

    //! Example doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that
        V(r) is continuous at the cutoff
        \note There is no need to check if rsq < rcutsq in this method.
        Cutoff tests are performed in PotentialPair.

        \return True if they are evaluated or false if they are not because
        we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq && lj1 != 0)
            {
            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar r6inv = r2inv * r2inv * r2inv;

            force_divr = r2inv * r6inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2);
            pair_eng = r6inv * (lj1 * r6inv - lj2);

            if (rsq < rminsq)
                {
                pair_eng += e_shift;
                }
            else
                {
                force_divr *= lam;
                pair_eng *= lam;
                }
            
            if (energy_shift)
                {
                Scalar rcut2inv = Scalar(1.0) / rcutsq;
                Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                pair_eng -= lam * rcut6inv * (lj1 * rcut6inv - lj2);
                }
            return true;
            }
        else
            return false;
        }

    //! Example doesn't eval LRC integrals
    DEVICE Scalar evalPressureLRCIntegral()
        {
        if (rcutsq == 0)
            {
            return Scalar(0.0);
            }
        // lj1 = 4.0 * epsilon * pow(sigma,12.0)
        // lj2 = 4.0 * epsilon * pow(sigma,6.0);
        // The complete integral is as follows
        // -\int_{r_{c}}^{\infty} g_{ij}(r) r \frac{d}{dr}\bigg(E_{ij}(r)\bigg) r^{2} dr
        // which evaluates to
        // 4 \varepsilon \sigma^{12} (\frac{4}{3 r_{c}^{9}}) - ...
        // 4 \varepsilon \sigma^{6} (\frac{2}{r_{c}^{3}})
        Scalar rcut3inv = fast::pow(rcutsq, -1.5);
        Scalar rcut9inv = rcut3inv * rcut3inv * rcut3inv;
        return lam * (lj1 * Scalar(4.0) / Scalar(3.0) * rcut9inv - lj2 * Scalar(2.0) * rcut3inv);
        }

    //! Example doesn't eval LRC integrals
    DEVICE Scalar evalEnergyLRCIntegral()
        {
        if (rcutsq == 0)
            {
            return Scalar(0.0);
            }
        // Note that lj1 and lj2 are defined above.
        // lj1 = 4.0 * epsilon * pow(sigma,12.0)
        // lj2 = 4.0 * epsilon * pow(sigma,6.0);
        // The complete integral is as follows
        // \int_{r_{c}}^{\infty} g_{ij}(r) E_{ij}(r) r^{2} dr
        // which evaluates to
        // 4 \varepsilon \sigma^{12} (\frac{1}{9 r_{c}^{9}}) - ...
        // 4 \varepsilon \sigma^{6} (\frac{1}{3 r_{c}^{3}})
        Scalar rcut3inv = fast::pow(rcutsq, -1.5);
        Scalar rcut9inv = rcut3inv * rcut3inv * rcut3inv;
        return lam * (lj1 / Scalar(9.0) * rcut9inv - lj2 / Scalar(3.0) * rcut3inv);
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("ashbaugh_pair");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar lj1;      //!< Stored epsilon from the constructor
    Scalar lj2;  //!< Stored sigma_6 from the constructor
    Scalar lam;  //!< Stored lam from the constructor
    Scalar rminsq;
    Scalar e_shift;
    };

    }  // end namespace md
    }  // end namespace hoomd

#endif // __PAIR_EVALUATOR_ASHBAUGH_H__
