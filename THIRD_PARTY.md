# Third-party sources

The release fetches upstream implementations separately. Their respective
copyrights and distribution terms continue to apply.

| Resource | Pinned revision / definition | Use |
|---|---|---|
| [AutoPSO](https://github.com/EMI-Group/AutoPSO) | `d99bbcdd944d6697fd3466d51c9d97a9fd6d1133` | Particle and outer-PSO update implementation |
| [EvoX](https://github.com/EMI-Group/evox) | `95d9bda6fe315d74f543579150b0708ade8bec5b` | PSO, CSO, CLPSO, SLPSOGS and SLPSOUS |
| [MetaBox](https://github.com/MetaEvo/MetaBox) | `5565a281ce4e03efc1f6a41230507d5b66637ace` | Five learned optimizers |
| [CEC2022 resources](https://github.com/P-N-Suganthan/2022-SO-BO) | `Python-CEC2022.zip` at `de20505283b76ec6bf17a2e8fc6052e655830691` | Fixed transformation files and independent Python reference |
| [CEC2011 resources](https://github.com/P-N-Suganthan/CEC-2011--Real_World_Problems) | Technical-report FM and signed radar equations | Independently implemented engineering objectives |

The AutoPSO core is verified against Git blob
`0e0158e6c58f335f363af92a4f1fc5f4b4aa697f` before loading. Its source is not
redistributed in this repository. The CEC Python archive at the pinned revision
contains the same reference implementation used in the study.

The MetaBox patch changes only the sample-cost alignment, evaluation accounting
and boundary-repair lines documented in `REPRODUCING.md`. The upstream BSD
three-clause license is retained in `patches/MetaBox_LICENSE`.
EvoX is distributed under its upstream GPL-3.0-or-later terms.

CoRePSO's CEC-derived objectives deliberately retain continuous F4 and apply the
Schaffer, hybrid-segment and arithmetic-ordering corrections described in the
manuscript. Radar includes both signs of the technical-report cosine sums.
These definition choices are part of the experimental protocol.
