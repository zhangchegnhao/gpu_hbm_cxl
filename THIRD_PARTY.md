# Third-party software plan

The project does not embed third-party source code. The `cycle-v0` and
`cycle-v1` timing backends fetch Ramulator 2.1 as an external pinned dependency
and apply project-owned overlays during the build.

| Project | Planned use | License | Pinned revision |
|---|---|---|---|
| Ramulator 2.1 | DRAM/PIM command timing core | MIT | `b30320bc9385b708e86b67ebb9f48858cc66d798` |
| ATTACC simulator | Reference for Ramulator extension and wrapper interfaces | MIT | `c60005143a6b492d7ef83231723386478b59a506` |

Code copied or substantially derived from either project must retain its
upstream copyright and MIT license notice. The Python system simulator in
`src/sieve_replay` is an independent implementation based on the published
Sieve execution model.
