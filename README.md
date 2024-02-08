# Algorithm ResNet

![ResNet50 Model Architecture](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*tH9evuOFqk8F41FG.png)

- This is the algorithm developed by Team ResNet for the Justraigs challenge.
- It uses ResNet50 pretrained algorithm for this image classification task.

In this challenge, participants are tasked with analyzing the fundus images and assigning each image to one of two classes: "referable glaucoma" or "no referable glaucoma". "Referable glaucoma" refers to eyes where the fundus image exhibits signs or features indicative of glaucoma that require further examination or referral to a specialist. In this case, visual field damage is expected. On the other hand, "no referable glaucoma" refers to cases where the fundus image does not show significant indications of glaucoma and does not require immediate referral. Very early disease, in which visual field damage is not yet expected, were also classified as ‘"no referable glaucoma".

In addition to the referable glaucoma classification, participants were further instructed to perform multi-label classification for ten additional features related to glaucoma. These features are specific characteristics or abnormalities that may be present in the fundus images of glaucoma patients. The multi-label classification task involves assigning relevant labels to each fundus image based on the presence or absence of these specific features. These additional features provide more detailed information about the specific characteristics observed in the fundus images of "referable glaucoma" cases.

- The task is in two parts:

### Task 1: Referral performance
Binary classification of referable glaucoma and no referable glaucoma

 - No referable glaucoma (NRG)
 - Referable glaucoma (RG)

### Task 2: Justification performance
Multi-label classification of ten additional features

 - Appearance neuroretinal rim superiorly (ANRS)
 - Appearance neuroretinal rim inferiorly (ANRI)
 - Retinal nerve fiber layer defect superiorly (RNFLDS)
 - Retinal nerve fiber layer defect inferiorly (RNFLDI)
 - Baring circumlinear vessel superiorly (BCLVS)
 - Baring circumlinear vessel inferiorly (BCLVI)
 - Nasalisation of vessel trunk (NVT)
 - Disc hemorrhages (DH)
 - Laminar dots (LD)
 - Large cup (LC)

 (Task 2 is only on those images that were graded as referable glaucoma)