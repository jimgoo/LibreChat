install:
	npm ci
	npm run frontend
	# now you can run `npm run frontend:dev` and `npm run backend:dev`

mongo:
	mkdir -p ~/mongo-libre-chat
	mongod --dbpath ~/mongo-libre-chat

search:
	# matches MEILI_MASTER_KEY key in .env
	~/Downloads/meilisearch-macos-apple-silicon --master-key DrhYf7zENyR6AlUCKmnz0eYASOQdl6zxH7s7MKFSfFCt

run-frontend:
	# runs on http://localhost:3080
	npm run frontend:dev

run-backend:
	npm run backend:dev

rsync:
	 rsync -av --progress LibreChat LibreChat-1 --exclude node_modules