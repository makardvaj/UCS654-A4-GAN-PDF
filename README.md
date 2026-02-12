# Learning Probability Density Functions using GANs

This repository contains an implementation of a Generative Adversarial Network (GAN) designed to learn the unknown probability density function (PDF) of a transformed random variable using only data samples

## 1. Objective
The goal is to implicitly model the probability density of a transformed variable $z$ using a Generator network, without assuming any analytical or parametric forms like Gaussian or exponential distributions.

## 2. Methodology

### Data Transformation (Step 1)
The $NO_2$ concentration feature ($x$) from the India Air Quality dataset was transformed into a new variable $z$ using the function $z = Tr(x) = x + a_r \cdot \sin(b_r \cdot x)$

**Parameters Used:**
Based on University Roll Number $r = 102316037$ :--
* **$a_r$** = $0.5 \times (r \pmod 7) = 1.5$ 
* **$b_r$** = $0.3 \times (r \pmod 5 + 1) = 0.8999999999999999$ 

### GAN Architecture (Step 2)
The GAN was designed to learn the distribution of $z$ solely from its samples :--
* **Generator ($G$):** Implicitly models the PDF by transforming error noise $error \sim N(0,1)$ into fake samples $z_f$.
* **Discriminator ($D$):** Distinguishes between real samples $z$ and fake samples $z_f = G(error)$.

### Density Estimation (Step 3)
The PDF is approximated using Kernel Density Estimation (KDE) on the generated samples.

## 3. Results & PDF Estimation
After training, the Generator was used to produce a large number of samples, and the PDF was estimated using Kernel Density Estimation (KDE).

### PDF Plot
![PDF Comparision](https://github.com/makardvaj/UCS654-A4-GAN-PDF/blob/main/Comparision%20of%20Real%20and%20GAN-Generate%20PDF.png)

## 4. Observations
Based on the results obtained, the following observations were made regarding the model's performance :--

* **Mode Coverage:** The GAN successfully captured the underlying modes of the transformed distribution, indicating that the generator learned the periodic nature introduced by the sine transformation.
* **Training Stability:** The competition between the generator and discriminator reached a stable equilibrium, avoiding common issues like vanishing gradients or extreme mode collapse.
* **Quality of Generated Distribution:** The KDE of the generated samples ($z_f$) closely aligns with the actual distribution of $z$, proving the GAN's ability to learn complex PDFs without parametric assumptions.

---

### Important Compliance Notes
* The model learned the distribution **only** from samples of $z$.
* No parametric PDF (Gaussian, etc.) was assumed during the training process.
