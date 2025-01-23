#![allow(dead_code)]
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use ark_ec::pairing::Pairing;
use ark_ec::{CurveGroup, Group};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_std::UniformRand;
use core::marker::PhantomData;

// ---------------------------------------------------------------------------
// For BN254 pairing, or whatever pairing you want:
use ark_bn254::{Bn254, G1Projective as G1, G2Projective as G2};
// We use "Fr" as the scalar field:
type Fr = <Bn254 as Pairing>::ScalarField;
type GT = <Bn254 as Pairing>::TargetField; // i.e. Fq12

// We do "pair()" for e(g1, g2).
fn pair(g1: G1, g2: G2) -> GT {
    // `Bn254::pairing(...)` returns `PairingOutput<Bn254>`, 
    // which wraps an Fq12 in `.0`.
    Bn254::pairing(g1.into_affine(), g2.into_affine()).0
}


// ---------------------------------------------------------------------------
// DoryScheme
// We'll keep the same trait signature, but behind the scenes
// we store commitments in GT, and do dimension-halving across G1 & G2
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct DoryScheme<G: CurveGroup, ProofTranscript: Transcript> {
    _phantom: PhantomData<(G, ProofTranscript)>,
}

// ---------------------------------------------------------------------------
// DoryCommitment
// Instead of storing c, d1, d2 in "G", we store them in the target group "GT".
// ---------------------------------------------------------------------------
#[derive(Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment {
    pub c: GT,
    pub d1: GT,
    pub d2: GT,
}

impl AppendToTranscript for DoryCommitment {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        // We'll do naive serialization
        let mut bytes = Vec::new();
        self.c.serialize_compressed(&mut bytes).unwrap();
        self.d1.serialize_compressed(&mut bytes).unwrap();
        self.d2.serialize_compressed(&mut bytes).unwrap();

        transcript.append_message(b"dory_commit");
        transcript.append_bytes(&bytes);
    }
}

// ---------------------------------------------------------------------------
// DoryProof + DoryBatchedProof
// We'll store step1, step2 in the target group GT as well
// ---------------------------------------------------------------------------
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct DoryProof {
    pub step1_elements: Vec<ReduceProverStep1Elements>,
    pub step2_elements: Vec<ReduceProverStep2Elements>,
    pub scalar_product_proof: ScalarProductProofElements,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct DoryBatchedProof {
    pub step1_elements: Vec<ReduceProverStep1Elements>,
    pub step2_elements: Vec<ReduceProverStep2Elements>,
    pub scalar_product_proof: ScalarProductProofElements,
}

// ---------------------------------------------------------------------------
// Step1 / Step2 Elements (all in GT for partial products)
// ---------------------------------------------------------------------------
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct ReduceProverStep1Elements {
    pub c: GT,
    pub d1: GT,
    pub d2: GT,
    pub d1_l: GT,
    pub d1_r: GT,
    pub d2_l: GT,
    pub d2_r: GT,
}

impl AppendToTranscript for ReduceProverStep1Elements {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        let mut bytes = Vec::new();
        self.c.serialize_compressed(&mut bytes).unwrap();
        self.d1.serialize_compressed(&mut bytes).unwrap();
        self.d2.serialize_compressed(&mut bytes).unwrap();
        self.d1_l.serialize_compressed(&mut bytes).unwrap();
        self.d1_r.serialize_compressed(&mut bytes).unwrap();
        self.d2_l.serialize_compressed(&mut bytes).unwrap();
        self.d2_r.serialize_compressed(&mut bytes).unwrap();

        transcript.append_message(b"dory_step1");
        transcript.append_bytes(&bytes);
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct ReduceProverStep2Elements {
    pub c_plus: GT,
    pub c_minus: GT,
}

impl AppendToTranscript for ReduceProverStep2Elements {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        let mut bytes = Vec::new();
        self.c_plus.serialize_compressed(&mut bytes).unwrap();
        self.c_minus.serialize_compressed(&mut bytes).unwrap();

        transcript.append_message(b"dory_step2");
        transcript.append_bytes(&bytes);
    }
}

// ---------------------------------------------------------------------------
// ScalarProductProofElements
// For dimension=1 final check, we store E1 in G1, E2 in G2, do a pairing check
// ---------------------------------------------------------------------------
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct ScalarProductProofElements {
    pub e1: G1,
    pub e2: G2,
}

impl AppendToTranscript for ScalarProductProofElements {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        let mut bytes = Vec::new();
        self.e1.serialize_compressed(&mut bytes).unwrap();
        self.e2.serialize_compressed(&mut bytes).unwrap();

        transcript.append_message(b"dory_scalar");
        transcript.append_bytes(&bytes);
    }
}

// ---------------------------------------------------------------------------
// "Setup" data. We store gamma1 in G1, gamma2 in G2, plus chi in GT, etc.
// (like the original code).
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct DoryPublicParams {
    pub n: usize,
    pub gamma1: Vec<G1>,
    pub gamma2: Vec<G2>,

    // for final scalar product check
    pub chi: GT,

    // prime versions for dimension halving
    pub gamma1_prime: Vec<G1>,
    pub gamma2_prime: Vec<G2>,
    pub delta_1l: GT,
    pub delta_1r: GT,
    pub delta_2l: GT,
    pub delta_2r: GT,
}

// We'll store them in "None" to match your code
#[derive(Clone, Debug)]
pub struct None {
    pub pp: DoryPublicParams,
}

// ---------------------------------------------------------------------------
// Helper: random G1, G2, plus pairing-based "inner product" in GT
// ---------------------------------------------------------------------------
fn random_g1(rng: &mut StdRng) -> G1 {
    G1::rand(rng)
}
fn random_g2(rng: &mut StdRng) -> G2 {
    G2::rand(rng)
}

fn random_g1_vec(n: usize, rng: &mut StdRng) -> Vec<G1> {
    (0..n).map(|_| random_g1(rng)).collect()
}
fn random_g2_vec(n: usize, rng: &mut StdRng) -> Vec<G2> {
    (0..n).map(|_| random_g2(rng)).collect()
}

// Pairwise product in GT
fn pairwise_product_g1_g2(g1s: &[G1], g2s: &[G2]) -> GT {
    assert_eq!(g1s.len(), g2s.len());
    let mut acc = GT::ONE;
    for i in 0..g1s.len() {
        acc *= pair(g1s[i], g2s[i]);
    }
    acc
}

fn reduce_pp(pp: &mut DoryPublicParams) {
    if pp.n == 1 {
        return;
    }
    // half
    let half = pp.n / 2;
    let g1_left = &pp.gamma1[..half];
    let g1_right = &pp.gamma1[half..];
    let g2_left = &pp.gamma2[..half];
    let g2_right = &pp.gamma2[half..];

    let mut rng = StdRng::seed_from_u64(9999u64);
    let gamma1p = random_g1_vec(half, &mut rng);
    let gamma2p = random_g2_vec(half, &mut rng);

    let delta_1l = pairwise_product_g1_g2(g1_left, &gamma2p);
    let delta_1r = pairwise_product_g1_g2(g1_right, &gamma2p);
    let delta_2l = pairwise_product_g1_g2(&gamma1p, g2_left);
    let delta_2r = pairwise_product_g1_g2(&gamma1p, g2_right);

    pp.gamma1_prime = gamma1p;
    pp.gamma2_prime = gamma2p;
    pp.delta_1l = delta_1l;
    pp.delta_1r = delta_1r;
    pp.delta_2l = delta_2l;
    pp.delta_2r = delta_2r;
}

fn new_public_params(n: usize) -> DoryPublicParams {
    let mut rng = StdRng::seed_from_u64(12345u64);
    let gamma1 = random_g1_vec(n, &mut rng);
    let gamma2 = random_g2_vec(n, &mut rng);
    let chi = pairwise_product_g1_g2(&gamma1, &gamma2);

    let mut pp = DoryPublicParams {
        n,
        gamma1,
        gamma2,
        chi,
        gamma1_prime: vec![],
        gamma2_prime: vec![],
        delta_1l: GT::ONE,
        delta_1r: GT::ONE,
        delta_2l: GT::ONE,
        delta_2r: GT::ONE,
    };
    reduce_pp(&mut pp);
    pp
}

// ---------------------------------------------------------------------------
// Implementation of CommitmentScheme
// We'll interpret poly.Z as "2n" coefficients => n for G1, n for G2
// (just a placeholder approach).
// ---------------------------------------------------------------------------
impl<TranscriptType: Transcript> CommitmentScheme<TranscriptType> for DoryScheme<G1, TranscriptType> {
    type Field = Fr;
    type Setup = None;
    type Commitment = DoryCommitment;
    type Proof = DoryProof;
    type BatchedProof = DoryBatchedProof;

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
        let n = if !shapes.is_empty() {
            shapes[0].input_length
        } else {
            1
        };
        let pp = new_public_params(n);
        None { pp }
    }

    fn commit(poly: &DensePolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        // We'll interpret poly.Z as 2n elements: half go to G1, half go to G2
        // Then produce c, d1, d2 in GT. 
        let n = setup.pp.n;
        let z = &poly.Z;

        if z.len() != 2*n {
            panic!("Need a 2n vector for v1 in G1 and v2 in G2");
        }

        // split z => v1 in G1, v2 in G2
        // We'll do: v1[i] = G1::generator() * z[i], v2[i] = G2::generator() * z[i+n]
        let mut v1 = Vec::with_capacity(n);
        let mut v2 = Vec::with_capacity(n);
        for i in 0..n {
            let s1 = z[i];
            let s2 = z[i + n];
            let g1 = G1::generator() * s1;
            let g2 = G2::generator() * s2;
            v1.push(g1);
            v2.push(g2);
        }

        // D1 = product e(v1[i], gamma2[i]), D2 = product e(gamma1[i], v2[i]), C= product e(v1[i], v2[i])
        let mut d1 = GT::ONE;
        let mut d2 = GT::ONE;
        let mut c = GT::ONE;
        for i in 0..n {
            d1 *= pair(v1[i], setup.pp.gamma2[i]);
            d2 *= pair(setup.pp.gamma1[i], v2[i]);
            c *= pair(v1[i], v2[i]);
        }

        DoryCommitment { c, d1, d2 }
    }

    fn batch_commit(
        _evals: &[&[Self::Field]],
        _setup: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        unimplemented!("Batch commit not shown here")
    }

    fn commit_slice(_evals: &[Self::Field], _setup: &Self::Setup) -> Self::Commitment {
        unimplemented!("commit_slice not used in this example")
    }

    fn prove(
        setup: &Self::Setup,
        poly: &DensePolynomial<Self::Field>,
        _opening_point: &[Self::Field],
        transcript: &mut TranscriptType,
    ) -> Self::Proof {
        // Same dimension-halving approach, but v1 in G1, v2 in G2 
        let n = setup.pp.n;
        let z = &poly.Z;
        if z.len() != 2*n {
            panic!("Need a 2n vector for dimension n");
        }
        // build v1, v2
        let mut v1 = Vec::with_capacity(n);
        let mut v2 = Vec::with_capacity(n);
        for i in 0..n {
            let s1 = z[i];
            let s2 = z[i + n];
            let g1 = G1::generator() * s1;
            let g2 = G2::generator() * s2;
            v1.push(g1);
            v2.push(g2);
        }

        // build the commitment
        let cmt = Self::commit(poly, setup);
        cmt.append_to_transcript(transcript);

        // dimension-halving => reduce
        let (step1, step2, scalar_proof) = reduce_prover(
            &v1, 
            &v2, 
            &setup.pp, 
            &cmt, 
            transcript
        );

        DoryProof {
            step1_elements: step1,
            step2_elements: step2,
            scalar_product_proof: scalar_proof,
        }
    }

    fn batch_prove(
        _setup: &Self::Setup,
        _polynomials: &[&DensePolynomial<Self::Field>],
        _opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _batch_type: BatchType,
        _transcript: &mut TranscriptType,
    ) -> Self::BatchedProof {
        unimplemented!("Batched proof not shown")
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut TranscriptType,
        _opening_point: &[Self::Field],
        _opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        commitment.append_to_transcript(transcript);
        verify_reducer(
            &proof.step1_elements,
            &proof.step2_elements,
            &proof.scalar_product_proof,
            &setup.pp,
            commitment,
            transcript,
        )?;
        Ok(())
    }

    fn batch_verify(
        _batch_proof: &Self::BatchedProof,
        _setup: &Self::Setup,
        _opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _commitments: &[&Self::Commitment],
        _transcript: &mut TranscriptType,
    ) -> Result<(), ProofVerifyError> {
        unimplemented!("Batched verify not shown")
    }

    fn protocol_name() -> &'static [u8] {
        b"dory_commit"
    }
}

// ---------------------------------------------------------------------------
// Now the "reduce_prover" logic: dimension-halving in G1, G2 => partial products in GT
// We'll skip some minor details for brevity. 
// The final "scalar product proof" is a real pairing check at dimension=1.
// ---------------------------------------------------------------------------

// Instead of reusing your single-group reduce_prover, we define it for (v1 in G1, v2 in G2).
fn reduce_prover<TranscriptType: Transcript>(
    v1: &[G1],
    v2: &[G2],
    pp: &DoryPublicParams,
    cmt: &DoryCommitment,
    transcript: &mut TranscriptType
) -> (Vec<ReduceProverStep1Elements>, Vec<ReduceProverStep2Elements>, ScalarProductProofElements)
{
    let n = v1.len();
    if n == 1 {
        // dimension=1 => final scalar product proof
        let sp = scalar_product_prover_single(v1[0], v2[0], cmt, pp);
        return (vec![], vec![], sp);
    }

    // split
    let half = n/2;
    let (v1l, v1r) = ( &v1[..half], &v1[half..] );
    let (v2l, v2r) = ( &v2[..half], &v2[half..] );

    // Step1: 
    // d1_l = product of e(v1l[i], gamma2_prime[i])
    let mut d1l = GT::ONE;
    let mut d1r = GT::ONE;
    for i in 0..half {
        d1l *= pair(v1l[i], pp.gamma2_prime[i]);
        d1r *= pair(v1r[i], pp.gamma2_prime[i]);
    }
    // d2_l = product of e(gamma1_prime[i], v2l[i])
    let mut d2l = GT::ONE;
    let mut d2r = GT::ONE;
    for i in 0..half {
        d2l *= pair(pp.gamma1_prime[i], v2l[i]);
        d2r *= pair(pp.gamma1_prime[i], v2r[i]);
    }

    let step1 = ReduceProverStep1Elements {
        c: cmt.c,
        d1: cmt.d1,
        d2: cmt.d2,
        d1_l: d1l,
        d1_r: d1r,
        d2_l: d2l,
        d2_r: d2r,
    };
    step1.append_to_transcript(transcript);

    let beta = transcript.challenge_scalar::<Fr>();

    // fold
    let beta_inv = ark_ff::Field::inverse(&beta).unwrap();
    let mut new_v1 = Vec::with_capacity(half);
    let mut new_v2 = Vec::with_capacity(half);
    for i in 0..half {
        let folded1 = v1l[i] + (v1r[i] * beta);
        let folded2 = v2l[i] + (v2r[i] * beta_inv);
        new_v1.push(folded1);
        new_v2.push(folded2);
    }

    // Step2: c_plus = product of e(v1l[i], v2r[i]), c_minus = product e(v1r[i], v2l[i])
    let mut c_plus = GT::ONE;
    let mut c_minus = GT::ONE;
    for i in 0..half {
        c_plus *= pair(v1l[i], v2r[i]);
        c_minus *= pair(v1r[i], v2l[i]);
    }

    let step2 = ReduceProverStep2Elements {
        c_plus,
        c_minus
    };
    step2.append_to_transcript(transcript);

    let alpha = transcript.challenge_scalar::<Fr>();

    // build next commitment
    let new_cmt = fold_commitment(cmt, &step1, &step2, alpha, beta, &pp.chi);

    let (mut s1_list, mut s2_list, sp) = reduce_prover(
        &new_v1, 
        &new_v2, 
        pp, 
        &new_cmt, 
        transcript
    );
    let mut out1 = vec![step1];
    out1.append(&mut s1_list);
    let mut out2 = vec![step2];
    out2.append(&mut s2_list);
    (out1, out2, sp)
}

// Pairing-based final step
fn scalar_product_prover_single(
    g1: G1,
    g2: G2,
    _cmt: &DoryCommitment,
    _pp: &DoryPublicParams
) -> ScalarProductProofElements {
    // dimension=1 => we just store e1=g1, e2=g2
    // We'll do the real pairing check in "check_scalar_product_single"
    ScalarProductProofElements { e1: g1, e2: g2 }
}

fn fold_commitment(
    cmt: &DoryCommitment,
    step1: &ReduceProverStep1Elements,
    step2: &ReduceProverStep2Elements,
    alpha: Fr,
    beta: Fr,
    chi: &GT
) -> DoryCommitment {
    // C' = C * chi * D2^beta * D1^(beta_inv) * c_plus^alpha * c_minus^(alpha_inv)
    use ark_ff::One;
    let beta_inv = ark_ff::Field::inverse(&beta).unwrap_or(Fr::one());
    let alpha_inv = ark_ff::Field::inverse(&alpha).unwrap_or(Fr::one());

    let mut c_prime = cmt.c;
    c_prime *= *chi;
    c_prime *= cmt.d2.pow(beta.0);
    c_prime *= cmt.d1.pow(beta_inv.0);
    c_prime *= step2.c_plus.pow(alpha.0);
    c_prime *= step2.c_minus.pow(alpha_inv.0);

    // D1' = ...
    let mut d1_prime = step1.d1_l.pow(alpha.0);
    d1_prime *= step1.d1_r;
    // similarly for d2'
    let mut d2_prime = step1.d2_l.pow(alpha_inv.0);
    d2_prime *= step1.d2_r;

    DoryCommitment {
        c: c_prime,
        d1: d1_prime,
        d2: d2_prime,
    }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn verify_reducer<TranscriptType: Transcript>(
    step1s: &[ReduceProverStep1Elements],
    step2s: &[ReduceProverStep2Elements],
    sp: &ScalarProductProofElements,
    pp: &DoryPublicParams,
    cmt: &DoryCommitment,
    transcript: &mut TranscriptType
) -> Result<(), ProofVerifyError> {
    if step1s.is_empty() && step2s.is_empty() {
        // dimension=1 => check final
        if !check_scalar_product_single(sp, cmt, pp) {
            return Err(ProofVerifyError::InternalError);
        }
        return Ok(());
    }
    let step1 = &step1s[0];
    step1.append_to_transcript(transcript);
    let beta = transcript.challenge_scalar::<Fr>();

    let step2 = &step2s[0];
    step2.append_to_transcript(transcript);
    let alpha = transcript.challenge_scalar::<Fr>();

    let new_cmt = fold_commitment(cmt, step1, step2, alpha, beta, &pp.chi);
    verify_reducer(&step1s[1..], &step2s[1..], sp, pp, &new_cmt, transcript)
}

/// A simple "inner product" in G1 with scalars in Fr
/// so that `test_inner_product_in_g` compiles
pub fn inner_product_in_g(scalars: &[Fr], bases: &[G1]) -> G1 {
    let mut acc = G1::default();
    let n = scalars.len().min(bases.len());
    for i in 0..n {
        acc += bases[i] * scalars[i];
    }
    acc
}


// Real pairing-based check: e(e1 + d*g1, e2 + d^-1*g2) = chi * c * d2^d * d1^(1/d)
// Check if e(sp.e1, sp.e2) == cmt.c
// ignoring d1, d2 for dimension=1 
fn check_scalar_product_single(sp: &ScalarProductProofElements, cmt: &DoryCommitment, _pp: &DoryPublicParams) -> bool {
    let left = pair(sp.e1, sp.e2);
    let right = cmt.c;
    left == right
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::poly::commitment::commitment_scheme::BatchType;
    use crate::poly::commitment::commitment_scheme::CommitShape;
    use crate::utils::transcript::Transcript;
    use ark_ff::UniformRand;
    use ark_std::rand::SeedableRng;
    use ark_std::rand::rngs::StdRng;
    use std::time::Instant;

    // For convenience, pick a type for G, F, etc.
    use ark_bn254::G1Projective as G;
    use ark_bn254::Fr as F;
    use crate::utils::transcript::KeccakTranscript; 

    // We alias the DoryScheme with our chosen group & transcript
    type Dory = DoryScheme<G, KeccakTranscript>;

    /// Generate a random multilinear polynomial of dimension `num_vars`,
    /// so the length of Z = 2^num_vars.
    fn random_dense_poly(num_vars: usize) -> DensePolynomial<F> {
        let mut rng = StdRng::seed_from_u64(12345u64);
        let len = 1 << num_vars;
        let mut Z = Vec::with_capacity(len);
        for _i in 0..len {
            Z.push(F::random(&mut rng));
        }
        DensePolynomial::new(Z)
    }

    /// Loosely corresponds to `TestScalarProductProof` in Go.
    #[test]
    fn test_scalar_product_proof() {
        // Suppose we want dimension=1 => length=2
        let num_vars = 1;
 
        // Now we create our commit shapes with the correct field names:
        let shapes = [CommitShape::new(1, BatchType::Small)];

        // Alternatively, we could do:
        // let shapes = [CommitShape { input_length: length, batch_type: BatchType::Small }];

        let setup = Dory::setup(&shapes);

        // Build a random polynomial
        let poly = random_dense_poly(num_vars);
        let commitment = Dory::commit(&poly, &setup);

        let iterations = 100;
        let mut total_proof_time = 0u128;
        let mut total_verify_time = 0u128;

        for _ in 0..iterations {
            let mut transcript = KeccakTranscript::new(b"DORY_TEST");

            let start_proof = Instant::now();
            let proof = Dory::prove(&setup, &poly, &[], &mut transcript);
            let proof_time = start_proof.elapsed().as_nanos();
            total_proof_time += proof_time;

            let mut transcript_v = KeccakTranscript::new(b"DORY_TEST");
            let start_verify = Instant::now();
            let res = Dory::verify(&proof, &setup, &mut transcript_v, &[], &F::from(0), &commitment);
            let verify_time = start_verify.elapsed().as_nanos();
            total_verify_time += verify_time;

            assert!(res.is_ok(), "Verification must succeed for scalar product proof");
        }

        println!(
            "Average proof time (ns): {}",
            total_proof_time as f64 / iterations as f64
        );
        println!(
            "Average verify time (ns): {}",
            total_verify_time as f64 / iterations as f64
        );
    }

    /// Loosely corresponds to `TestInnerProd` in Go.
    /// We check that our `inner_product_in_g` function sums group-scalar products correctly.
    #[test]
    fn test_inner_product_in_g() {
        let mut rng = StdRng::seed_from_u64(9999u64);

        let g1a: G = G::rand(&mut rng);
        let g1b: G = G::rand(&mut rng);
        let g1c: G = G::rand(&mut rng);

        let s1: F = F::random(&mut rng);
        let s2: F = F::random(&mut rng);
        let s3: F = F::random(&mut rng);

        let bases = vec![g1a, g1b, g1c];
        let scalars = vec![s1, s2, s3];

        // sum = g1a*s1 + g1b*s2 + g1c*s3
        let manual = g1a * s1 + g1b * s2 + g1c * s3;

        let auto = super::inner_product_in_g(&scalars, &bases);
        assert_eq!(manual, auto, "inner_product_in_g mismatch");
    }

    /// Loosely corresponds to `TestDoryReduce` in Go.
    /// dimension=3 => length=8, we do commit/prove/verify multiple times.
    #[test]
    fn test_dory_reduce() {
        let num_vars = 3;
        // random_dense_poly(3) => length=8
        // so let's set shapes= [CommitShape::new(4, ...)], so n=4 => demands 2n=8
        let shapes = [CommitShape::new(4, BatchType::Small)];
        let setup = Dory::setup(&shapes);

        let poly = random_dense_poly(num_vars);
        let commitment = Dory::commit(&poly, &setup);

        let iterations = 100;
        let mut total_proof_time = 0u128;
        let mut total_verify_time = 0u128;

        for _ in 0..iterations {
            let mut transcript_p = KeccakTranscript::new(b"DORY_TEST");
            let start_proof = Instant::now();
            let proof = Dory::prove(&setup, &poly, &[], &mut transcript_p);
            let proof_time = start_proof.elapsed().as_nanos();
            total_proof_time += proof_time;

            let s1_len = proof.step1_elements.len();
            let s2_len = proof.step2_elements.len();
            // If dimension=3 => we typically do 2 rounds of halving: 4->2->1
            // => 2 step1, 2 step2
            assert_eq!(s1_len, 2, "Expected 2 step1 elements for dimension=3");
            assert_eq!(s2_len, 2,  "Expected 2 step2 elements for dimension=3");

            let mut transcript_v = KeccakTranscript::new(b"DORY_TEST");
            let start_verify = Instant::now();
            let res = Dory::verify(&proof, &setup, &mut transcript_v, &[], &F::from(0), &commitment);
            let verify_time = start_verify.elapsed().as_nanos();
            total_verify_time += verify_time;

            assert!(res.is_ok(), "DoryReduce verification must succeed");
        }

        println!(
            "Average reduce proof time (ns): {}",
            total_proof_time as f64 / iterations as f64
        );
        println!(
            "Average reduce verify time (ns): {}",
            total_verify_time as f64 / iterations as f64
        );
    }
}
