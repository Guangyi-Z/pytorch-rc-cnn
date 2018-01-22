# pytorch-rc-cnn
Reading Comprehension on CNN dataset in PyTorch

## Reference

Chen, Danqi, Jason Bolton, and Christopher D. Manning. "A thorough examination of the cnn/daily mail reading comprehension task." arXiv preprint arXiv:1606.02858 (2016).

## Performance

```
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch0: 0.320846075433
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch1: 0.373598369011
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch2: 0.374617737003
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch3: 0.363914373089
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch4: 0.405963302752
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch5: 0.427370030581
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch6: 0.425840978593
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch7: 0.434760448522
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch8: 0.441896024465
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch9: 0.446738022426
acc-lr0.05-momentum0.0-trainszNone-embsz100-hidsz128-epoch10: 0.445973496432
```

## TODO

* Vectorization. So far minibatch==1 only, which leads to badly poor estimation of gradients
