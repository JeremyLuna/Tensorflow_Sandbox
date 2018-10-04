// TODO: make angular datasets,
// and one where y doesnt matter or
// ellipse, to test weighted norms

function simple_dataset(){
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

function cluster_dataset(){
  data = [];
  labels = [];

  x_coord = randf(-4.0, 4.0);
  y_coord = randf(-4.0, 4.0);
  for (x = 0; x < 10; x++){
    data.push([randn(x_coord, 0.3), randn(y_coord, 0.3)]);
    labels.push(0);
  }

  for (x = 0; x < 15; x++){
    data.push([randf(-4.0, 4.0), randf(-4.0, 4.0)]);
    labels.push(1);
  }

  return {'data': data, 'labels': labels};
}

function circle_dataset(){
  data = [];
  labels = [];

  for (x = 0.0; x <= 2*Math.PI; x += .1*Math.PI){
    data.push([.5*Math.cos(x)+1.0, .5*Math.sin(x)+1.0]);
    labels.push(0);
  }

  for (x = 0; x < 10; x++){
    data.push([randf(-4.0, 4.0), randf(-4.0, 4.0)]);
    labels.push(1);
  }

  return {'data': data, 'labels': labels};
}

function telephone_pole_dataset(){
  data = [];
  labels = [];

  for (x = 0.0; x <= 2*Math.PI; x += .1*Math.PI){
    data.push([.5*Math.cos(x)+1.0, .5*Math.sin(x)+1.0]);
    labels.push(0);
  }
  for (x = 0.0; x <= 2*Math.PI; x += .1*Math.PI){
    data.push([.5*Math.cos(x)-1.0, .5*Math.sin(x)-1.0]);
    labels.push(0);
  }

  for (x = 0.0; x <= 2*Math.PI; x += .1*Math.PI){
    data.push([Math.cos(x)+1.0, Math.sin(x)+1.0]);
    labels.push(1);
  }
  for (x = 0.0; x <= 2*Math.PI; x += .1*Math.PI){
    data.push([Math.cos(x)-1.0, Math.sin(x)-1.0]);
    labels.push(1);
  }

  return {'data': data, 'labels': labels};
}

function spiral_dataset(){
  data = [];
  labels = [];

  for (x = 0.0; x <= 3*Math.PI; x += .1*Math.PI){
    data.push([0.5*x*Math.cos(x), 0.5*x*Math.sin(x)]);
    labels.push(0);
  }

  for (x = 0.0; x <= 3*Math.PI; x += .1*Math.PI){
    data.push([0.5*x*Math.cos(x+Math.PI), 0.5*x*Math.sin(x+Math.PI)]);
    labels.push(1);
  }

  return {'data': data, 'labels': labels};
}

function random_dataset(){
  data = [];
  labels = [];

  for (x = 0; x < 10; x++){
    data.push([randf(-4.0, 4.0), randf(-4.0, 4.0)]);
    labels.push(0);
  }
  for (x = 0; x < 10; x++){
    data.push([randf(-4.0, 4.0), randf(-4.0, 4.0)]);
    labels.push(1);
  }

  return {'data': data, 'labels': labels};
}

function angular_dataset(){
  data = [];
  labels = [];

  function get_angle(x, y){
    var angle = Math.atan2(y, x);
    var degrees = 180*angle/Math.PI;
    return (360+Math.round(degrees))%360;
  }


  for (dot = 0; dot < 20; dot++){
    x = randf(-1.0, 1.0)
    y = randf(-1.0, 1.0)
    data.push([x, y]);
    if (get_angle(x, y) > 45 && get_angle(x, y) < 135){
      labels.push(0);
    }else{
      labels.push(1);
    }
  }

  return {'data': data, 'labels': labels};
}
