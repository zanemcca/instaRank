var Db = require('mongodb').Db,
    Server = require('mongodb').Server,
    assert = require('assert'),
    MongoClient = require('mongodb').MongoClient;


var Views, Clicks, UpVotes, DownVotes;

var mongoclient = new MongodbClient(new Server('localhost', 27017), {native_parser: true});
mongoclient.open(function(err, mongoclient) {
  var db = mongoclient.db('interactions');

  // Create a test collection
  db.collection('view', function(err, collection) {
    Views = collection;
    db.collection('click', function(err, collection) {
      Clicks = collection;
      db.collection('upVote', function(err, collection) {
        UpVotes = collection;
        db.collection('downVote', function(err, collection) {
          DownVotes = collection;
        });
      });
    });
  });
});

/*
var mapReduce(collection, map, reduce, cb) {
  // Peform the map reduce
  collection.mapReduce(map, reduce, {out: {replace : 'tempCollection', readPreference : 'secondary'}}, function(err, collection) {
    if(err) { 
      console.error(err);
      return cb(err); 
    }
    cb(err, collection);
  });
};

function execute() {
  // Map function
  var map = function() { emit(this.viewableId, this.type); };
  // Reduce function
  var reduce = function(k,vals) {
    return (vals.length > 0);
  };

  //TODO MapReduce all clicks upVotes and downVotes
  //TODO Split clicks into groups
  mapReduce(Clicks, map, reduce, function(err, clicks) {
    mapReduce(UpVotes, map, reduce, function(err, upvotes) {
      mapReduce(DownVotes, map, reduce, function(err, downVotes) {
      });
    });
  });
}
*/

function execute() {
  Views.find().sort({ created: -1 }).function(err, views) {
    if(err) {
      return console.error(err);
    }

    for(var i in views) {
      funcs.push(processView.bind(null, views[i]));
    }

    async.parallel(funcs, function(err, res) {
      if(err) {
        return console.error(err);
      }
    });
  };
}

function processView(view, cb) {
  var x = {};
  var y = {};

  getY(function() {
    //TODO Get x
    //TODO Get the viewable item
    //TODO Get all children of the viewable item
    //init x for time t = 0;

    //TODO
  });

  function getY(cb) {
    Clicks.find({ viewableId: view.id }, function(err, clicks) {
      if(err) {
        return console.error(err);
      }

      for(var i in clicks) {
        y[clicks[i].type] = 1;
      }

      UpVotes.find({ viewableId: view.id }, function(err, upvotes) {
        if(err) {
          return console.error(err);
        }
        if(upvotes.length === 1) {
          y.upvote = 1;
        } else if(upvotes.length > 1) {
          console.error('More than one upvote in a view!');
        }

        DownVotes.find({ viewableId: view.id }, function(err, downvotes) {
          if(err) {
            return console.error(err);
          }

          if(downvotes.length === 1) {
            y.downvote = 1;
          } else if(downvotes.length > 1) {
            console.error('More than one downvote in a view!');
          }

          cb();
        });
      });
    });
  }
}
