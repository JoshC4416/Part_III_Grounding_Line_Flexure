# Part_III_Grounding_Line_Flexure
Implementation of numerical approach for solving 1D ice sheet-shelf flexure profiles (currently produces numerical artefacts)

This repository contain the code used in the Part III project on Grounding Line Flexure completed for the QCES course.
The current version of the code is not yet deemed well-conditioned, as it produces numerical artefacts for realistic Young's modulus values for ice, however outlines the approach used, which shows promise with more pre-conditioning, or with the use of numerical continuation.
Much of the numerical approach implemented here is builds off of the method outlined by Butler & Neufeld (2023 [Unpublished]).
