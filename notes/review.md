Dear Sitao Min, Hafiz Asif, Xinyue, Jaideep Vaidya,

We have reached a decision regarding your submission to Journal of Statistical Software, #5850 “FedImpute: A Framework for Distributed Missing Data Imputation Research with Python”.

Unfortunately we have to tell you that we decided to reject in its current state.

However, we encourage you to resubmit the article after detailed revision. Please check the journal management system for any additional feedback: https://www.jstatsoft.org/index.php/jss/authorDashboard/submission/5850. We expect you to provide a point-by-point response to this feedback in your revision.

Your sincerely,

The editorial team

JSS Administrator
editor@jstatsoft.org



------------------------------------------------------
Reviewer A:
Recommendation: Choose One

------------------------------------------------------


Rate the overall quality of the submission:

Running the replication materials:

Briefly describe your computational setup:

Running a new or modified example:

Are the help files in the package:

Detailed comments regarding the manuscript, software, and replication materials:

------------------------------------------------------



------------------------------------------------------
Reviewer C:

Please see my comments below.

Recommendation: Revisions Required

------------------------------------------------------


there are also some important shortcomings that affect reproducibility and long-term usability. In particular, the public GitHub repository currently lacks basic Continuous Integration (CI) checks. This is a significant limitation, as it makes it difficult to ensure that the codebase remains functional across evolving Python environments and dependencies.

I was able to reproduce the experiments on Ubuntu 24.04 with Python 3.12.3 (without Docker), after pinning scipy to 1.15.3 due to a deprecation issue in later versions. However, another reviewer encountered issues on a more recent Linux system with Python 3.14, suggesting the setup may not be robust across environments.


### Major Recommendation (Blocking)

- While providing a Docker container or fixed dependency versions is a valid approach for deployment, incorporating a minimal CI pipeline (e.g., automated tests across one or multiple Python versions) would greatly improve the reliability and impact of the project. Given the complexity of the Python ecosystem, such safeguards are increasingly essential for research software.

### Major Recommendation

- The paper would benefit from a clearer discussion of the intended deployment setting of FedImpute. In particular, can the framework handle genuinely distributed datasets located on remote clients, or is it currently limited to simulating distributed settings from locally stored data? Relatedly, does FedImpute support any client-server deployment mode, or is it primarily a research framework for controlled local experimentation? Clarifying this point would help readers better understand the practical scope of the framework and its relevance to real-world federated environments.

 

### Minor Comments and Questions

- The motivation for using t-SNE in the experiments is not justified. Are there specific advantages in this context compared to alternative visualization or dimensionality reduction methods?

- The installation command pins a specific version (p.17 Section 4.1 `pip install fedimpute==0.2.0`). It would be helpful to clarify whether this is intended for reproducibility of the paper’s results or general usage, and whether users are encouraged to install newer versions when available.

### Typos and Editorial Issues

- p.10 – Bullet point “Missing Data Simulation – Missing Data Heterogeneity”: missing colon.

- p.17 – Line 13: “IID” is not consistently capitalized (appears as “iid”).

- p.28 – “FedImput” should be “FedImpute”.

 


-------------------------------

Journal of Statistical Software http://www.jstatsoft.org/