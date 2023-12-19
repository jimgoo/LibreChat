install:
	npm ci

mongo:
	mkdir -p ~/mongo-libre-chat
	mongod --dbpath ~/mongo-libre-chat

search:
	# matches MEILI_MASTER_KEY key in .env
	~/Downloads/meilisearch-macos-apple-silicon --master-key DrhYf7zENyR6AlUCKmnz0eYASOQdl6zxH7s7MKFSfFCt

run:
	# runs on http://localhost:3080
	npm run backend
