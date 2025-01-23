use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use crate::{
    field::JoltField,
    poly::dense_mlpoly::DensePolynomial,
    utils::{
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, Transcript},
    },
};
use super::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};

#[derive(Clone)]
pub struct DoryCommitScheme<F: JoltField, ProofTranscript: Transcript> {
    _marker: PhantomData<(F, ProofTranscript)>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Default, Debug, PartialEq)]
pub struct DoryCommitment<F: JoltField> {
    pub commitment_value: Vec<F>,
}

impl<F: JoltField> AppendToTranscript for DoryCommitment<F> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_message(b"dory_commitment");
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct DoryProof<F: JoltField> {
    pub witness: Vec<F>,
    pub eval: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct DoryBatchProof<F: JoltField> {
    pub batched_witness: Vec<F>,
    pub evals: Vec<F>,
}

impl<F, ProofTranscript> DoryCommitScheme<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub fn open(
        _setup: &Vec<F>,
        poly: &DensePolynomial<F>,
        point: &[F],
    ) -> DoryProof<F> {
        let eval = poly.evaluate(point);
        let witness = compute_witness(poly, point);
        DoryProof { witness, eval }
    }

    pub fn batch_open(
        _setup: &Vec<F>,
        polys: &[&DensePolynomial<F>],
        point: &[F],
    ) -> DoryBatchProof<F> {
        let batched_eval = polys.iter().map(|poly| poly.evaluate(point)).collect();
        let batched_witness = polys
            .iter()
            .flat_map(|poly| compute_witness(poly, point))
            .collect();
        DoryBatchProof {
            batched_witness,
            evals: batched_eval,
        }
    }
}

impl<F, ProofTranscript> CommitmentScheme<ProofTranscript> for DoryCommitScheme<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    type Field = F;
    type Setup = Vec<F>;
    type Commitment = DoryCommitment<F>;
    type Proof = DoryProof<F>;
    type BatchedProof = DoryBatchProof<F>;

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
        shapes
            .iter()
            .map(|shape| {
                assert!(
                    shape.input_length.is_power_of_two(),
                    "Input length must be a power of 2"
                );
                F::from_u64(shape.input_length as u64).unwrap()
            })
            .collect()
    }

    fn commit(poly: &DensePolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        let commitment_value = poly
            .evals_ref()
            .iter()
            .zip(setup.iter())
            .map(|(coeff, param)| *coeff * *param)
            .collect();
        println!(
            "Commitment: {:?}, Poly: {:?}, Setup: {:?}",
            commitment_value,
            poly.evals_ref(),
            setup
        );
        DoryCommitment { commitment_value }
    }
    

    fn batch_commit(
        evals: &[&[Self::Field]],
        setup: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        evals
            .iter()
            .map(|poly_evals| {
                let commitment_value = poly_evals
                    .iter()
                    .zip(setup.iter())
                    .map(|(coeff, param)| *coeff * *param)
                    .collect();
                DoryCommitment { commitment_value }
            })
            .collect()
    }

    fn commit_slice(evals: &[Self::Field], setup: &Self::Setup) -> Self::Commitment {
        let commitment_value = evals
            .iter()
            .zip(setup.iter())
            .map(|(coeff, param)| *coeff * *param)
            .collect();
        DoryCommitment { commitment_value }
    }

    fn prove(
        _setup: &Self::Setup,
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let eval = poly.evaluate(opening_point);
        let witness = compute_witness(poly, opening_point);
        transcript.append_message(b"prove_dory");
        DoryProof { witness, eval }
    }

    fn batch_prove(
        _setup: &Self::Setup,
        polys: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _batch_type: BatchType,
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        let evals: Vec<_> = polys.iter().map(|poly| poly.evaluate(opening_point)).collect();
        let batched_witness: Vec<_> = polys
            .iter()
            .flat_map(|poly| compute_witness(poly, opening_point))
            .collect();
        transcript.append_message(b"batch_prove_dory");
        DoryBatchProof {
            batched_witness,
            evals,
        }
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        _transcript: &mut ProofTranscript,
        _opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let reconstructed_commitment = DoryCommitment {
            commitment_value: proof
                .witness
                .iter()
                .zip(setup.iter())
                .map(|(w, s)| *w * *s)
                .collect(),
        };
    
        println!(
            "Reconstructed Commitment: {:?}, Original Commitment: {:?}",
            reconstructed_commitment, commitment
        );
    
        if reconstructed_commitment == *commitment && proof.eval == *opening {
            Ok(())
        } else {
            Err(ProofVerifyError::InternalError)
        }
    }
    
    

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        setup: &Self::Setup,
        _opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        for ((eval, commitment), opening) in batch_proof
            .evals
            .iter()
            .zip(commitments.iter())
            .zip(openings.iter())
        {
            let reconstructed_commitment = DoryCommitment {
                commitment_value: setup
                    .iter()
                    .map(|s| *eval * *s)
                    .collect(),
            };
    
            println!(
                "Batch Reconstructed Commitment: {:?}, Original Commitment: {:?}, Eval: {:?}, Opening: {:?}",
                reconstructed_commitment, commitment, eval, opening
            );
    
            if &reconstructed_commitment != *commitment || eval != opening {
                return Err(ProofVerifyError::InternalError);
            }
        }
        Ok(())
    }
    
    
    fn protocol_name() -> &'static [u8] {
        b"DoryCommitScheme"
    }
}

fn compute_witness<F: JoltField>(
    poly: &DensePolynomial<F>,
    point: &[F],
) -> Vec<F> {
    let mut witness = poly.evals_ref().to_vec();
    for &p in point {
        println!("Processing point: {:?}", p);
        for j in (1..witness.len()).rev() {
            let original = witness[j - 1];
            let added = witness[j] * p;
            witness[j - 1] += added;
            println!(
                "witness[{}] updated: {:?} (original: {:?}, added: {:?})",
                j - 1,
                witness[j - 1],
                original,
                added
            );
        }
    }
    println!("Final witness: {:?}", witness);
    witness
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_std::UniformRand;

    #[test]
    fn test_dory_single_open_and_verify() {
        type TestDoryCommitScheme = DoryCommitScheme<Fr, KeccakTranscript>;


        // Ensure polynomial evaluations are a power of 2
    
        // Ensure polynomial evaluations are a power of 2
        let poly = DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
        println!("Polynomial: {:?}", poly.evals_ref());
        
        let point = vec![Fr::from(2)];
        println!("Point: {:?}", point);
    
        let eval = poly.evaluate(&point);
        println!("Expected Evaluation: {:?}", eval);
    
        let setup = vec![Fr::from(3), Fr::from(2), Fr::from(1), Fr::from(4)];
        println!("Setup: {:?}", setup);
    
        let commitment = TestDoryCommitScheme::commit(&poly, &setup);
        println!("Commitment: {:?}", commitment);
    
        let mut transcript = KeccakTranscript::new(b"DoryTest");
        let proof = TestDoryCommitScheme::prove(&setup, &poly, &point, &mut transcript);
        println!("Proof: {:?}", proof);
    
        assert!(TestDoryCommitScheme::verify(
            &proof,
            &setup,
            &mut transcript,
            &point,
            &eval,
            &commitment
        )
        .is_ok());
    }
    
    #[test]
    fn test_dory_batch_open_and_verify() {
        type TestDoryCommitScheme = DoryCommitScheme<Fr, KeccakTranscript>;

        // Create univariate polynomials
        let polys = vec![
            DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]),
            DensePolynomial::new(vec![Fr::from(5), Fr::from(6), Fr::from(7), Fr::from(8)]),
        ];

        // Single point for univariate polynomials
        let point = vec![Fr::from(2)];
        let setup = vec![Fr::from(3), Fr::from(2), Fr::from(1), Fr::from(4)];

        let commitments: Vec<_> = polys
            .iter()
            .map(|poly| TestDoryCommitScheme::commit(poly, &setup))
            .collect();

        let evals: Vec<_> = polys.iter().map(|poly| poly.evaluate(&point)).collect();
        let mut transcript = KeccakTranscript::new(b"DoryBatchTest");
        let proofs = TestDoryCommitScheme::batch_prove(
            &setup,
            &polys.iter().collect::<Vec<_>>(),
            &point,
            &evals,
            BatchType::Big,
            &mut transcript,
        );

        assert!(TestDoryCommitScheme::batch_verify(
            &proofs,
            &setup,
            &point,
            &evals,
            &commitments.iter().collect::<Vec<_>>(),
            &mut transcript,
        )
        .is_ok());
    }
}
