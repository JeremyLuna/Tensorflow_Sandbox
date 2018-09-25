function original_dataset(){
  data = [];
  labels = [];
  data.push([-0.4326,  1.1909]); labels.push(1);
  data.push([ 3.0   ,  4.0   ]); labels.push(1);
  data.push([ 0.1253, -0.0376]); labels.push(1);
  data.push([ 0.2877,  0.3273]); labels.push(1);
  data.push([-1.1465,  0.1746]); labels.push(1);
  data.push([ 1.8133,  1.0139]); labels.push(0);
  data.push([ 2.7258,  1.0668]); labels.push(0);
  data.push([ 1.4117,  0.5593]); labels.push(0);
  data.push( [4.1832,  0.3044]); labels.push(0);
  data.push([ 1.8636,  0.1677]); labels.push(0);
  data.push([ 0.5   ,  3.2   ]); labels.push(1);
  data.push([ 0.8   ,  3.2   ]); labels.push(1);
  data.push([ 1.0   , -2.2   ]); labels.push(1);
  return {'data': data, 'labels': labels};
}
